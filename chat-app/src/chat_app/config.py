"""AI Orchestrator Service Configuration using Pydantic Settings.

Provides centralized configuration for the chat-app service including:
- LLM settings (model, temperature, max_tokens)
- Service URLs (Redis, ChromaDB)
- API keys and external integrations

Configuration is loaded from environment variables and .env files using
Pydantic Settings. The @lru_cache decorator ensures a single settings
instance is shared across the application.

Last Grunted: 02/03/2026 03:15:00 PM PST
"""
import os
from typing import Optional, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

# Environment-based configuration for NIM (NVIDIA Inference Microservice) support
# When set, routes LLM calls through NIM endpoint instead of OpenAI API
OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
NIM_MODEL_NAME: Optional[str] = os.getenv("NIM_MODEL_NAME")


class ServiceSettings(BaseSettings):
    """Core service configuration for chat-app.

    Pydantic Settings class that loads configuration from environment
    variables and .env files. Provides typed access to all service settings.

    Settings are grouped by category:
    - Agent settings: workspace, matplotlib backend
    - LLM settings: default model, temperature, max tokens
    - Redis settings: host, port, database
    - ChromaDB settings: persist directory
    - External APIs: Perplexity

    Example:
        >>> settings = get_settings()
        >>> print(settings.default_model)
        "gpt-4o"
        >>> print(settings.get_redis_url())
        "redis://localhost:6379/0"

    Last Grunted: 02/03/2026 03:45:00 PM PST
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Agent settings
    agent_workspace: str = Field(default="artifacts", description="Directory for agent file operations")
    matplotlib_backend: str = Field(default="Agg", description="Matplotlib backend for headless operation")
    
    # Default LLM settings
    default_model: str = Field(default="gpt-4o", description="Default LLM model")
    default_temperature: float = Field(default=0.0, description="Default LLM temperature")
    default_max_tokens: Optional[int] = Field(default=None, description="Default max tokens")
    
    # Redis settings (for caching/sessions)
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    
    # ChromaDB settings
    chroma_persist_dir: str = Field(default="./data/chroma", description="ChromaDB persistence directory")
    
    # Perplexity API
    perplexity_api_key: Optional[str] = Field(default=None, env="PERPLEXITY_API_KEY")
    perplexity_model: str = Field(default="sonar-reasoning", description="Perplexity model for web search")
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL in standard format.

        Returns:
            str: Redis URL in format "redis://host:port/db"

        Last Grunted: 02/03/2026 03:45:00 PM PST
        """
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache()
def get_settings() -> ServiceSettings:
    """Get cached singleton settings instance.

    Uses @lru_cache to ensure a single ServiceSettings instance is created
    and reused across the application lifetime. The settings are loaded
    from environment variables and .env files on first call.

    Returns:
        ServiceSettings: Cached configuration instance with all service settings.

    Note:
        To reload settings (e.g., after env changes), call get_settings.cache_clear()
        before calling get_settings() again.

    Last Grunted: 02/03/2026 03:15:00 PM PST
    """
    return ServiceSettings()
