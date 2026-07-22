from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    OLLAMA_BASE_URL: str
    CHROMA_PERSIST_DIR: str

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
