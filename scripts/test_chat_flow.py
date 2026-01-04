"""Simulate a chat flow locally using LocalProvider for embedding+generation and InMemory DB.

Flow:
 - Configure settings for LOCAL embedding and generation
 - Index patients (in-memory)
 - Search with a sample Arabic query
 - Build prompt with contexts and ask generation client
 - Print the generated reply
"""
from helpers.config import get_settings
from stores.LLM.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from controllers.PatientController import PatientController


def build_prompt(contexts, question):
    header = (
        "أجب باللغة العربية وبشكل مهني. استخدم فقط سياق المرضى الآتي. اذكر دوماً أن الاستشارة النهائية للطبيب.")
    parts = [header, "\n"]
    for i, c in enumerate(contexts, start=1):
        parts.append(f"CONTEXT {i}: {c}\n")
    parts.append("\nQUESTION: " + question)
    return "\n".join(parts)


def extract_text_from_result(r):
    # r may have several nesting shapes depending on provider
    # Try common patterns
    if not r:
        return ""
    # pattern: {'payload': {'payload': {'text': ...}, 'metadata': ...}, 'score': ...}
    if isinstance(r, dict):
        p = r.get('payload')
        if isinstance(p, dict):
            inner = p.get('payload') or p.get('text')
            if isinstance(inner, dict):
                return inner.get('text') or ''
            if isinstance(inner, str):
                return inner
        # fallback: maybe payload itself is the text
        if isinstance(p, str):
            return p
        # direct text
        if 'text' in r:
            return r['text']
    return ''


if __name__ == '__main__':
    settings = get_settings()
    # configure for local test
    settings.EMBEDDING_BACKEND = 'LOCAL'
    settings.EMBEDDING_MODEL_ID = 'local'
    settings.EMBEDDING_MODEL_SIZE = 64
    settings.GENERATION_BACKEND = 'LOCAL'
    settings.GENERATION_MODEL_ID = 'local_gen'
    settings.VECTOR_DB_BACKEND = 'INMEMORY'

    llm_factory = LLMProviderFactory(settings)
    embedding_client = llm_factory.create(provider=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID, embedding_size=settings.EMBEDDING_MODEL_SIZE)

    generation_client = llm_factory.create(provider=settings.GENERATION_BACKEND)
    generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    vec_factory = VectorDBProviderFactory(settings)
    vec_provider = vec_factory.create(provider=settings.VECTOR_DB_BACKEND)
    vec_provider.connect()

    pc = PatientController()
    ok = pc.index_patients_to_qdrant(embedding_client=embedding_client, vector_db_provider=vec_provider, collection_name='patients')
    print('Indexed:', ok)

    question = 'ما خيارات العلاج للمريضات اللواتي لديهن ER Positive وتتعلق بالعلاج الكيماوي؟'
    results = pc.search_patients(query=question, embedding_client=embedding_client, vector_db_provider=vec_provider, collection_name='patients', top_k=3)

    contexts = []
    for r in results:
        t = extract_text_from_result(r.get('payload')) if isinstance(r, dict) else extract_text_from_result(r)
        if not t:
            # try other shapes
            t = extract_text_from_result(r)
        if t:
            contexts.append(t)

    prompt = build_prompt(contexts, question)
    print('\n--- PROMPT ---\n')
    print(prompt)
    print('\n--- BOT RESPONSE ---\n')
    resp = generation_client.generate_text(prompt)
    print(resp)

    vec_provider.disconnect()
