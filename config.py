from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field, field_validator
from typing import Literal
from functools import lru_cache




class ModelConfig(BaseSettings):
    provider: Literal["openai", "gemini"] = Field(
        validation_alias=AliasChoices("MODEL_PROVIDER", "provider")
    )
    model: str = Field(validation_alias=AliasChoices("MODEL_NAME", "model"))
    gemini_api_key: str
    base_url: str | None = Field(
        default=None, validation_alias=AliasChoices("MODEL_BASE_URL", "base_url")
    )
    temperature: float | None = Field(
        default=None, validation_alias=AliasChoices("MODEL_TEMPERATURE", "temperature")
    )
    max_tokens: int | None = Field(
        default=None, validation_alias=AliasChoices("MODEL_MAX_TOKENS", "max_tokens")
    )
    timeout: float | None = Field(
        default=60, validation_alias=AliasChoices("MODEL_TIMEOUT", "timeout")
    )
    max_retries: int = Field(
        default=2, validation_alias=AliasChoices("MODEL_MAX_RETRIES", "max_retries")
    )

    @field_validator("base_url", "temperature", "max_tokens", "timeout", mode="before")
    @classmethod
    def blank_optional_setting(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    
@lru_cache
def get_settings() -> ModelConfig:
    return ModelConfig()

settings = get_settings()