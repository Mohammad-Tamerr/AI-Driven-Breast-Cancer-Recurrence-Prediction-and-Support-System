from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    OPENAI_API_KEY: str

    FILE_ALLOWED_TYPES: list 
    MAX_FILE_SIZE_MB: int
    CHUNK_SIZE: int

    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str
    VECTOR_DB_DISTANCE_METHOD: str = None

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
