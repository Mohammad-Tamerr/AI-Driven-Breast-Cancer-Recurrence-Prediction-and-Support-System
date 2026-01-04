"""Script to index patients.json into Qdrant collection 'patients'"""
import logging
from helpers.config import get_settings
from stores.LLM.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from controllers.PatientController import PatientController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    settings = get_settings()

    if not settings.EMBEDDING_BACKEND:
        logger.error("EMBEDDING_BACKEND not configured in settings (.env)")
        return

    if not settings.VECTOR_DB_BACKEND:
        logger.error("VECTOR_DB_BACKEND not configured in settings (.env)")
        return

    llm_factory = LLMProviderFactory(settings)
    embedding_client = llm_factory.create(provider=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID, embedding_size=settings.EMBEDDING_MODEL_SIZE)

    vec_factory = VectorDBProviderFactory(settings)
    vec_provider = vec_factory.create(provider=settings.VECTOR_DB_BACKEND)
    vec_provider.connect()

    patient_controller = PatientController()
    ok = patient_controller.index_patients_to_qdrant(embedding_client=embedding_client, vector_db_provider=vec_provider, collection_name="patients")

    if ok:
        logger.info("✅ Patients indexed into Qdrant collection: patients")
    else:
        logger.error("❌ Indexing patients failed")

    vec_provider.disconnect()


if __name__ == "__main__":
    main()
