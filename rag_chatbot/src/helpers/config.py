from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    OPENAI_API_KEY: str

    FILE_ALLOWED_TYPES: list 
    MAX_FILE_SIZE_MB: int
    CHUNK_SIZE: int

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
