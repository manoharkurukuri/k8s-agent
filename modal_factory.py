from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings
from gemini_openai import GeminiChatOpenAI




def create_model():

    if settings.provider == "openai":
        model_class = GeminiChatOpenAI if settings.model.startswith("gemini-") else ChatOpenAI
        return model_class(
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
            base_url=settings.base_url,
            api_key=settings.gemini_api_key,
        )

    if settings.provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.model,
            api_key=settings.gemini_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
            base_url=settings.base_url,
        )

    raise ValueError(f"Unsupported provider: {settings.provider}")


model = create_model()
