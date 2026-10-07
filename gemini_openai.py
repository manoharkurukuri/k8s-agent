from copy import deepcopy
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from openai import BaseModel


_TOOL_METADATA = "gemini_openai_tool_metadata"


def _tool_metadata(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Avoid the special "index" key: LangChain merges indexed list entries and
    # concatenates strings, which would corrupt repeated opaque signatures.
    return [
        {
            "call_index": call.get("index", index),
            "call_id": call.get("id"),
            "extra_content": deepcopy(call.get("extra_content")),
        }
        for index, call in enumerate(tool_calls)
        if call.get("id") or call.get("extra_content")
    ]


class GeminiChatOpenAI(ChatOpenAI):
    """Preserve Gemini tool metadata over the OpenAI Chat Completions API."""

    use_responses_api: bool | None = False

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices", [])
        if generation is not None and choices:
            delta = choices[0].get("delta") or {}
            metadata = _tool_metadata(delta.get("tool_calls") or [])
            if metadata:
                generation.message.additional_kwargs[_TOOL_METADATA] = metadata
        return generation

    def _create_chat_result(
        self,
        response: dict | BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, response_dict["choices"], strict=True):
            metadata = _tool_metadata(choice["message"].get("tool_calls") or [])
            if metadata:
                generation.message.additional_kwargs[_TOOL_METADATA] = metadata
        return result

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "messages" not in payload:
            raise ValueError("Gemini OpenAI compatibility requires the Chat Completions API")
        messages = self._convert_input(input_).to_messages()
        for message, outgoing in zip(messages, payload["messages"], strict=True):
            if not isinstance(message, AIMessage):
                continue
            metadata = message.additional_kwargs.get(_TOOL_METADATA, [])
            ids_by_index = {
                entry["call_index"]: entry["call_id"]
                for entry in metadata
                if entry["call_id"]
            }
            calls_by_id = {
                call["id"]: call for call in outgoing.get("tool_calls", [])
            }
            for entry in metadata:
                if entry["extra_content"] is None:
                    continue
                call_id = entry["call_id"] or ids_by_index.get(entry["call_index"])
                if call_id not in calls_by_id:
                    raise ValueError("Cannot associate Gemini tool metadata with its tool call")
                calls_by_id[call_id]["extra_content"] = deepcopy(entry["extra_content"])
        return payload
