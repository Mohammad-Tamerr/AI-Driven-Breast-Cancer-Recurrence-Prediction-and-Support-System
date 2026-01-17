from fastapi import APIRouter, Depends, Request
from helpers.config import get_settings, Settings
from controllers.PatientController import PatientController
from pydantic import BaseModel
from fastapi.responses import JSONResponse


patients_router = APIRouter(
    prefix="/Rafeek/v1/patients",
    tags=["Patients"]
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class ChatRequest(BaseModel):
    question: str
    top_k: int = 3


@patients_router.post("/index")
def index_patients(request: Request, app_settings: Settings = Depends(get_settings)):
    """Trigger indexing of patients.json into vector DB
    Reuse the app-level embedding client and vector DB provider when available to avoid concurrent local Qdrant instances."""

    # Prefer app-level clients initialized at startup
    app = request.app

    embedding_client = getattr(app, 'embedding_client', None)
    if embedding_client is None and app_settings.EMBEDDING_BACKEND:
        from stores.LLM.LLMProviderFactory import LLMProviderFactory
        llm_factory = LLMProviderFactory(app_settings)
        embedding_client = llm_factory.create(provider=app_settings.EMBEDDING_BACKEND)
        if embedding_client and app_settings.EMBEDDING_MODEL_ID:
            embedding_client.set_embedding_model(model_id=app_settings.EMBEDDING_MODEL_ID, embedding_size=app_settings.EMBEDDING_MODEL_SIZE)

    vec_provider = getattr(app, 'vector_db_provider', None)
    created_local_vec = False
    if vec_provider is None and app_settings.VECTOR_DB_BACKEND:
        from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
        vec_factory = VectorDBProviderFactory(app_settings)
        vec_provider = vec_factory.create(provider=app_settings.VECTOR_DB_BACKEND)
        if vec_provider:
            vec_provider.connect()
            created_local_vec = True

    if not embedding_client or not vec_provider:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Embedding or Vector DB not configured"})

    pc = PatientController()
    ok = pc.index_patients_to_qdrant(embedding_client=embedding_client, vector_db_provider=vec_provider, collection_name="patients")

    if created_local_vec:
        vec_provider.disconnect()

    if ok:
        return JSONResponse(status_code=200, content={"status": "ok", "message": "Patients indexed"})
    else:
        return JSONResponse(status_code=500, content={"status": "error", "message": "Indexing failed"})


@patients_router.post("/search")
def search_patients(request: Request, req: SearchRequest, app_settings: Settings = Depends(get_settings)):
    # Use app-level clients if available to avoid opening multiple Qdrant clients
    app = request.app

    if not app_settings.EMBEDDING_BACKEND or not app_settings.VECTOR_DB_BACKEND:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Embedding or Vector DB not configured"})

    embedding_client = getattr(app, 'embedding_client', None)
    if embedding_client is None and app_settings.EMBEDDING_BACKEND:
        from stores.LLM.LLMProviderFactory import LLMProviderFactory
        llm_factory = LLMProviderFactory(app_settings)
        embedding_client = llm_factory.create(provider=app_settings.EMBEDDING_BACKEND)
        if embedding_client and app_settings.EMBEDDING_MODEL_ID:
            embedding_client.set_embedding_model(model_id=app_settings.EMBEDDING_MODEL_ID, embedding_size=app_settings.EMBEDDING_MODEL_SIZE)

    vec_provider = getattr(app, 'vector_db_provider', None)
    created_local_vec = False
    if vec_provider is None and app_settings.VECTOR_DB_BACKEND:
        from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
        vec_factory = VectorDBProviderFactory(app_settings)
        vec_provider = vec_factory.create(provider=app_settings.VECTOR_DB_BACKEND)
        if vec_provider:
            vec_provider.connect()
            created_local_vec = True

    if not embedding_client or not vec_provider:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Embedding or Vector DB not configured"})

    pc = PatientController()
    results = pc.search_patients(query=req.query, embedding_client=embedding_client, vector_db_provider=vec_provider, collection_name="patients", top_k=req.top_k)

    if created_local_vec:
        vec_provider.disconnect()

    # Add a user-friendly Arabic message depending on whether we have results
    if not results:
        message = "ما لقيتش حالات مشابهة في قاعدة البيانات. ممكن تجرب صياغة مختلفة للاستعلام أو أفهرس الحالات الآن—تحب أعمل الفهرسة؟"
    else:
        message = "لقيت حالات مشابهة — عايز أعرض الأفضل أو ألخص لك؟"

    return {"status": "ok", "results": results, "message": message}


@patients_router.get("/{patient_id}")
def get_patient(patient_id: str):
    pc = PatientController()
    p = pc.get_patient_by_id(patient_id)
    if not p:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Patient not found"})
    return p


@patients_router.get("/status")
def patients_status(request: Request, app_settings: Settings = Depends(get_settings)):
    """Return whether the `patients` collection exists and its size (number of records)."""
    app = request.app
    vec_provider = getattr(app, 'vector_db_provider', None)

    if vec_provider is None:
        return {"status": "ok", "collection_exists": False, "size": 0, "message": "Vector DB not configured"}

    try:
        exists = vec_provider.is_collection_existed("patients")
        info = vec_provider.get_collection_info("patients") if exists else {}
        # normalize size from different providers
        size = 0
        if isinstance(info, dict):
            size = info.get("size") or info.get("vectors_count") or info.get("points_count") or 0
        return {"status": "ok", "collection_exists": exists, "size": int(size)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@patients_router.post("/{patient_id}/chat")
def patient_chat(patient_id: str, request: Request, req: ChatRequest, app_settings: Settings = Depends(get_settings)):
    """Chat with the system about a specific patient. The response will be tailored using the patient's data."""
    app = request.app

    # Get generation client; fallback to None (LocalProvider will be used)
    generation_client = getattr(app, 'generation_client', None)
    embedding_client = getattr(app, 'embedding_client', None)
    vec_provider = getattr(app, 'vector_db_provider', None)

    pc = PatientController()
    resp = pc.chat_with_patient(patient_id=patient_id, question=req.question, generation_client=generation_client, embedding_client=embedding_client, vector_db_provider=vec_provider, top_k=req.top_k)

    if isinstance(resp, dict) and resp.get('error'):
        return JSONResponse(status_code=404, content={"status": "error", "message": resp.get('error')})

    return {"status": "ok", "answer": resp.get('answer'), "sources": resp.get('sources', []), "patient_id": patient_id}