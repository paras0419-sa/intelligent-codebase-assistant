from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Uses pydantic-settings which reads values in this priority order:
    1. Environment variables (highest priority)
    2. .env file
    3. Default values defined here (lowest priority)

    API keys use SecretStr so they are masked in logs/repr. To get the
    actual value, call .get_secret_value() explicitly — this forces you
    to be intentional about where secrets are exposed.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Treat empty string env vars (e.g. ANTHROPIC_API_KEY=) as unset
        env_ignore_empty=True,
    )

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    default_model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096

    # Controls how verbose the agent output is
    verbose: bool = False


# Module-level singleton. Every module imports the same instance.
# This is safe because settings are read-only after startup.
settings = Settings()
