"""Test indexing and searching with Local embeddings + InMemory DB in one process"""
from helpers.config import get_settings
from stores.LLM.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from controllers.PatientController import PatientController

settings = get_settings()
settings.EMBEDDING_BACKEND = "LOCAL"
settings.EMBEDDING_MODEL_ID = "local"
settings.EMBEDDING_MODEL_SIZE = 64
settings.VECTOR_DB_BACKEND = "INMEMORY"

llm_factory = LLMProviderFactory(settings)
embedding_client = llm_factory.create(provider=settings.EMBEDDING_BACKEND)
embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID, embedding_size=settings.EMBEDDING_MODEL_SIZE)

vec_factory = VectorDBProviderFactory(settings)
vec_provider = vec_factory.create(provider=settings.VECTOR_DB_BACKEND)
vec_provider.connect()

pc = PatientController()
ok = pc.index_patients_to_qdrant(embedding_client=embedding_client, vector_db_provider=vec_provider, collection_name="patients")
print("Indexed: ", ok)

results = pc.search_patients(query="ER Positive chemotherapy", embedding_client=embedding_client, vector_db_provider=vec_provider, collection_name="patients", top_k=3)
print("Search results:")
for r in results:
    print(r)

vec_provider.disconnect()
