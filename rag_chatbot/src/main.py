from fastapi import FastAPI
from contextlib import asynccontextmanager

from routes import base, data, test
from helpers.config import get_settings
from stores.LLM.LLMProviderFactory import LLMProviderFactory


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()
    print(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Validate API keys before creating providers
    if not settings.OPENAI_API_KEY:
        print("⚠️  Warning: OPENAI_API_KEY is not set")
    
    if not settings.COHERE_API_KEY:
        print("⚠️  Warning: COHERE_API_KEY is not set")
        
    if not settings.GEMINI_API_KEY:
        print("⚠️  Warning: GEMINI_API_KEY is not set")
    
    # Only initialize providers if API keys are available
    try:
        llm_provider_factory = LLMProviderFactory(settings)
        
        # generation client (can be OpenAI, Cohere, or Gemini)
        if settings.GENERATION_BACKEND:
            api_key_available = (
                (settings.GENERATION_BACKEND == "OPENAI" and settings.OPENAI_API_KEY) or
                (settings.GENERATION_BACKEND == "COHERE" and settings.COHERE_API_KEY) or
                (settings.GENERATION_BACKEND == "GEMINI" and settings.GEMINI_API_KEY)
            )
            
            if api_key_available:
                app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
                app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)
                print(f"✅ Generation client initialized: {settings.GENERATION_BACKEND}")
            else:
                app.generation_client = None
                print("⚠️  Generation client skipped - missing API key")
        else:
            app.generation_client = None

        # embedding client (can be OpenAI, Cohere, or Gemini)
        if settings.EMBEDDING_BACKEND:
            api_key_available = (
                (settings.EMBEDDING_BACKEND == "OPENAI" and settings.OPENAI_API_KEY) or
                (settings.EMBEDDING_BACKEND == "COHERE" and settings.COHERE_API_KEY) or
                (settings.EMBEDDING_BACKEND == "GEMINI" and settings.GEMINI_API_KEY)
            )
            
            if api_key_available:
                app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
                app.embedding_client.set_embedding_model(
                    model_id=settings.EMBEDDING_MODEL_ID,
                    embedding_size=settings.EMBEDDING_MODEL_SIZE
                )
                print(f"✅ Embedding client initialized: {settings.EMBEDDING_BACKEND}")
            else:
                app.embedding_client = None
                print("⚠️  Embedding client skipped - missing API key")
        else:
            app.embedding_client = None
            
    except Exception as e:
        print(f"❌ Error initializing LLM providers: {e}")
        app.generation_client = None
        app.embedding_client = None
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")


app = FastAPI(lifespan=lifespan)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(test.test_router)

