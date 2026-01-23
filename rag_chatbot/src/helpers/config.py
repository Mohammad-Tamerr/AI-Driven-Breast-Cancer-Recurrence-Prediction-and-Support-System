from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Basic app metadata
    APP_NAME: str = "mini-RAG"
    APP_VERSION: str = "0.1"
    
    # File handling defaults
    FILE_ALLOWED_TYPES: list = ["txt", "pdf", "md"]
    MAX_FILE_SIZE_MB: int = 10
    CHUNK_SIZE: int = 1000

    # Optional external resources
    MONGODB_URL: str = None
    MONGODB_DATABASE: str = None

    # LLM / embeddings
    GENERATION_BACKEND: str = None
    EMBEDDING_BACKEND: str = None

    # Vector DB
    VECTOR_DB_BACKEND: str = None
    VECTOR_DB_PATH: str = None
    VECTOR_DB_DISTANCE_METHOD: str = None

    # Provider keys
    OPENAI_API_KEY: str = None
    OPENAI_API_URL: str = None
    COHERE_API_KEY: str = None
    GEMINI_API_KEY: str = None

    # Model configuration
    GENERATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_MODEL_SIZE: int = None
    INPUT_DAFAULT_MAX_CHARACTERS: int = 1000
    GENERATION_DAFAULT_MAX_TOKENS: int = 1000
    GENERATION_DAFAULT_TEMPERATURE: float = 0.2

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
