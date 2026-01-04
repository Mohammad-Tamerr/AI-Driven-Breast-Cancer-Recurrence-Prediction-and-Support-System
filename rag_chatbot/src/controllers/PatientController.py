import json
import os
from pathlib import Path
from .BaseController import BaseController
import logging

logger = logging.getLogger(__name__)

class PatientController(BaseController):
    def __init__(self):
        super().__init__()
        # default patients file at repo root data/patients.json
        repo_root = Path(__file__).resolve().parents[3]
        self.default_path = os.path.join(repo_root, "data", "patients.json")

    def load_patients(self, path: str = None):
        path = path or self.default_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Patients file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_patient_by_id(self, patient_id: str, path: str = None):
        patients = self.load_patients(path=path)
        return next((p for p in patients if p.get("patient_id") == patient_id), None)

    def summarize_patient(self, patient: dict) -> str:
        biomarkers = ", ".join([f"{k}: {v}" for k, v in (patient.get("biomarkers") or {}).items()])
        treatments = "; ".join([f"{t.get('date')} - {t.get('type')} ({t.get('details')})" for t in (patient.get("treatments") or [])])
        summary = (
            f"Patient {patient.get('patient_id')} — age {patient.get('age')}, stage {patient.get('stage')}, "
            f"tumor: {patient.get('tumor_type')}. Biomarkers: {biomarkers}. Treatments: {treatments}. Follow-up: {patient.get('follow_up')}"
        )
        return summary

    def index_patients_to_qdrant(self, embedding_client, vector_db_provider, collection_name: str = "patients"):
        """Create collection and index patients as documents (payload contains full patient record)"""
        patients = self.load_patients()

        if not hasattr(embedding_client, "embed_text"):
            raise ValueError("Embedding client does not provide embed_text method")

        embedding_size = getattr(embedding_client, "embedding_size", None)
        if embedding_size is None:
            raise ValueError("Embedding client does not have embedding_size set")

        # create collection (reset if exists)
        vector_db_provider.create_collection(collection_name=collection_name, embedding_size=embedding_size, do_reset=True)

        texts = []
        vectors = []
        metadata = []
        record_ids = []

        for p in patients:
            text = self.summarize_patient(p)
            try:
                vec = embedding_client.embed_text(text, document_type="patient")
            except Exception as e:
                logger.error(f"Error embedding patient {p.get('patient_id')}: {e}")
                continue

            texts.append(text)
            vectors.append(vec)
            metadata.append({"patient_id": p.get("patient_id"), "patient_record": p})
            record_ids.append(p.get("patient_id"))

        # bulk insert
        ok = vector_db_provider.insert_many(collection_name=collection_name, texts=texts, vectors=vectors, metadata=metadata, record_ids=record_ids, batch_size=50)

        return ok

    def search_patients(self, query: str, embedding_client, vector_db_provider, collection_name: str = "patients", top_k: int = 5):
        if not query or not query.strip():
            return []

        # Ensure collection exists
        try:
            if not vector_db_provider.is_collection_existed(collection_name):
                logger.error(f"Collection {collection_name} not found")
                return []
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return []

        # Attempt to embed the query; fallback to LocalProvider on failure
        try:
            qvec = embedding_client.embed_text(query, document_type="query")
        except Exception as e:
            logger.error(f"Embedding query failed: {e}. Falling back to LocalProvider.")
            try:
                from stores.LLM.Providers.LocalProvider import LocalProvider
                fallback = LocalProvider()
                # try to set reasonable embedding size
                emb_size = getattr(embedding_client, 'embedding_size', 128) or 128
                fallback.set_embedding_model(model_id="local", embedding_size=emb_size)
                qvec = fallback.embed_text(query, document_type="query")
            except Exception as e2:
                logger.error(f"LocalProvider fallback also failed: {e2}")
                return []

        if not qvec:
            logger.error("Embedding returned empty vector")
            return []

        try:
            results = vector_db_provider.search_by_vector(collection_name=collection_name, vector=qvec, limit=top_k)
        except ValueError as ve:
            logger.error(f"Search error: {ve}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return []

        out = []
        for r in results:
            payload = r.payload if hasattr(r, 'payload') else r[0].payload if isinstance(r, (list, tuple)) else r
            # qdrant returns score as 'score' attribute
            score = getattr(r, 'score', None)
            out.append({"payload": payload, "score": score})

        return out

    def chat_with_patient(self, patient_id: str, question: str, generation_client, embedding_client=None, vector_db_provider=None, top_k: int = 3):
        """Answer a question about a specific patient using their data as context.
        - patient_id: identifier of the patient
        - question: user question (Arabic or English)
        - generation_client: LLM provider used for generation
        - embedding_client/vector_db_provider: optional - could be used for RAG augmentation
        Returns: dict {answer: str, sources: list}
        """
        # Load patient record
        p = self.get_patient_by_id(patient_id)
        if not p:
            return {"error": "Patient not found"}

        # Build patient context
        summary = self.summarize_patient(p)
        details = []
        if p.get("treatments"):
            details.append("Treatments: " + "; ".join([f"{t.get('date')}: {t.get('type')} ({t.get('details')})" for t in p.get('treatments')]))
        if p.get("notes"):
            details.append("Notes: " + p.get("notes"))

        context = summary + "\n" + "\n".join(details)

        # Optionally augment with RAG retrieved snippets if providers provided
        retrieved = []
        if embedding_client and vector_db_provider:
            try:
                # perform a scoped search and then filter to this patient_id
                qvec = None
                try:
                    qvec = embedding_client.embed_text(question, document_type="query")
                except Exception:
                    qvec = None

                if qvec:
                    results = vector_db_provider.search_by_vector(collection_name="patients", vector=qvec, limit=top_k)
                    # filter results to this patient
                    for r in results:
                        payload = r.payload if hasattr(r, 'payload') else r[0].payload if isinstance(r, (list, tuple)) else r
                        meta = payload.get('metadata') if isinstance(payload, dict) else None
                        pid = None
                        if isinstance(meta, list) and len(meta) and isinstance(meta[0], dict):
                            pid = meta[0].get('patient_id')
                        elif isinstance(meta, dict):
                            pid = meta.get('patient_id')

                        if pid == patient_id:
                            retrieved.append(payload)
            except Exception as e:
                logger.error(f"Error during RAG retrieval for patient chat: {e}")

        # detect language and build prompt accordingly (Arabic by default)
        import re

        def _detect_language(text: str) -> str:
            if not text or not text.strip():
                return 'ar'
            arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
            latin_chars = re.findall(r'[A-Za-z]', text)
            if len(latin_chars) > len(arabic_chars):
                return 'en'
            eng_keywords = ['surgery', 'operate', 'surgical', 'mastectomy', 'lumpectomy', 'fertility', 'palliative', 'follow-up', 'chemotherapy']
            if any(k in text.lower() for k in eng_keywords):
                return 'en'
            return 'ar'

        lang = _detect_language(question)

        if lang == 'en':
            header = (
                "You are a virtual medical assistant. Answer clearly and concisely in English, and rely only on the patient data below. "
                "Do not provide a substitute for professional medical advice."
            )
            instruct = "Answer briefly (3-5 sentences) and list any assumptions you made."
            rag_label = "Retrieved snippets from the patient's record:"
            info_label = "Patient info:"
            query_label = "Query:"
        else:
            header = (
                "أنت مساعد طبي افتراضي. أجب بلغة واضحة ومراعية، واستند فقط إلى بيانات المريضة الواردة أدناه. "
                "لا تقدم بديلاً عن رأي الطبيب؛ إنما قدم معلومات عامة وتوجيهات قابلة للنقاش مع الأخصائي."
            )
            instruct = "أجب باختصار (3-5 جمل)، واذكر أي افتراضات قمت بها."
            rag_label = "مقتطفات مستخرجة من سجلات المريضة:"
            info_label = "معلومات المريضة:"
            query_label = "الاستعلام:"

        prompt = f"{header}\n\n"
        prompt += (f"{info_label}\n{context}\n\n")
        prompt += (f"{query_label} {question}\n\n{instruct}")

        if retrieved:
            prompt += f"\n\n{rag_label}\n"
            for r in retrieved:
                text = r.get('text') if isinstance(r, dict) else str(r)
                prompt += f"- {text}\n"

        # Generate (try main generation client, then fallback to LocalProvider if result is None/empty)
        try:
            answer = None
            used_fallback = False

            if generation_client:
                try:
                    answer = generation_client.generate_text(prompt, chat_history=[], max_output_tokens=200, temperature=0.2)
                except Exception as e:
                    logger.error(f"Generation client failed: {e}")
                    answer = None

            if not answer:
                try:
                    from stores.LLM.Providers.LocalProvider import LocalProvider
                    lp = LocalProvider()
                    answer = lp.generate_text(prompt)
                    used_fallback = True
                except Exception as e:
                    logger.error(f"LocalProvider fallback failed: {e}")
                    answer = None

        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            return {"error": "Generation failed"}

        # intent detection + more specific rule-based fallback tailored to the question
        def _detect_intent(q: str) -> str:
            ql = (q or '').lower()
            if any(w in ql for w in ['surgery', 'operate', 'surgical', 'mastectomy', 'lumpectomy', 'جراح', 'عملية', 'استئصال']):
                return 'surgery'
            if any(w in ql for w in ['fertility', 'خصوب', 'حمل', 'خصوبة', 'oocyte', 'sperm']):
                return 'fertility'
            if any(w in ql for w in ['palliative', 'تلطيف', 'تخفيف', 'comfort']):
                return 'palliative'
            if any(w in ql for w in ['follow', 'متابعة', 'follow-up', 'checkup', 'متابعة']):
                return 'follow_up'
            if any(w in ql for w in ['protein', 'calorie', 'calories', 'nutrition', 'diet', 'food', 'بروتين', 'سعرات', 'تغذية', 'غذاء']):
                return 'nutrition'
            if any(w in ql for w in ['her2', 'er+', 'er positive', 'hormone', 'هرموني', 'HER2', 'ER']):
                return 'biomarker'
            return 'general'

        def _rule_based_answer(patient_obj, question_text, intent='general', lang='ar'):
            # gather patient attributes
            bm = patient_obj.get('biomarkers') or {}
            er = str(bm.get('ER') or '').lower()
            pr = str(bm.get('PR') or '').lower()
            her2 = str(bm.get('HER2') or '').lower()
            stage = str(patient_obj.get('stage') or '').lower()
            tumor = str(patient_obj.get('tumor_type') or '').lower()
            age = patient_obj.get('age')
            metas = patient_obj.get('metastasis_sites') or []
            prior = [t.get('type') for t in (patient_obj.get('treatments') or [])]
            brca = patient_obj.get('genetic_tests', {}).get('BRCA')
            comorbidities = [c.lower() for c in (patient_obj.get('comorbidities') or [])]

            ql = (question_text or '').lower()
            parts = []

            # tailored answers per intent
            if intent == 'surgery':
                # If metastatic or stage IV -> surgery less likely curative
                if 'iv' in stage or len(metas) > 0 or 'metastatic' in tumor:
                    if lang == 'en':
                        parts.append("This patient has metastatic disease or stage IV features, so systemic or palliative treatments usually take precedence; surgery is typically not curative in this setting. Discuss case in a multidisciplinary tumor board.")
                    else:
                        parts.append("المريضة تبدو بحالة نقيلية أو مرحلة IV، لذلك غالبًا ما تكون العلاجات الجهازية أو التلطيفية أولوية؛ الجراحة قد لا تكون علاجًا شافيًا في هذه الحالة. نوصي بمناقشة الحالة في اجتماع متعدد التخصصات.")
                else:
                    # If prior surgery exists, comment accordingly
                    if any('surgery' in t.lower() or 'mastectomy' in t.lower() or 'lumpectomy' in t.lower() for t in prior):
                        if lang == 'en':
                            parts.append("The patient has prior surgery; need to review margins, imaging, and response to any neoadjuvant therapy to decide on further surgical interventions.")
                        else:
                            parts.append("المريضة خضعت سابقًا لعملية؛ يجب مراجعة الهوامش والتصوير واستجابة العلاجات السابقة قبل اتخاذ قرار جراحي إضافي.")
                    else:
                        if lang == 'en':
                            parts.append("Surgical options (lumpectomy vs mastectomy) depend on tumour size, location, margins and patient preference; consult surgical oncology for imaging and biopsy correlation.")
                        else:
                            parts.append("خيارات الجراحة (جراحة حفظية مقابل استئصال كامل) تعتمد على حجم الورم وموقعه والهوامش وتفضيل المريضة؛ استشر جراح الأورام لمراجعة الصور والنتائج النسيجية.")

            elif intent == 'fertility':
                if age and age < 40:
                    if lang == 'en':
                        parts.append("Because the patient is young and may receive chemotherapy, consider urgent referral to fertility preservation (oocyte/embryo cryopreservation) before systemic therapy.")
                    else:
                        parts.append("بما أن المريضة شابة وقد تتلقى العلاج الكيميائي، يفضّل إحالتها سريعًا لحفظ الخصوبة (تجميد بويضات/أجنة) قبل بدء العلاج الجهازِي.")
                else:
                    if lang == 'en':
                        parts.append("Discuss fertility preservation options with a specialist; some options may be less effective after certain systemic therapies.")
                    else:
                        parts.append("ناقشي خيارات حفظ الخصوبة مع أخصائي؛ بعض الخيارات قد تكون أقل فاعلية بعد بعض العلاجات الجهازية.")
                # mention male-specific
                if patient_obj.get('name', '').lower().startswith('مريض') or patient_obj.get('name', '').lower().startswith('patient'):
                    if lang == 'en':
                        parts.append("For male patients, sperm cryopreservation is an option before systemic therapy.")
                    else:
                        parts.append("بالنسبة للمرضى الذكور، تجميد النطاف خيار قبل العلاج الجهازِي.")

            elif intent == 'palliative':
                if 'iv' in stage or len(metas) > 0 or 'metastatic' in tumor:
                    if lang == 'en':
                        parts.append("Integration of palliative care alongside oncologic treatment is appropriate; focus on symptom control and quality of life."
                                     )
                    else:
                        parts.append("من المناسب دمج الرعاية التلطيفية مع العلاج الأورامِي؛ التركيز يكون على السيطرة على الأعراض وتحسين نوعية الحياة.")
                else:
                    if lang == 'en':
                        parts.append("Palliative approaches may be considered for symptom control, but curative-intent treatments may still be relevant depending on stage.")
                    else:
                        parts.append("قد تُستخدم استراتيجيات تلطيفية للسيطرة على الأعراض، لكن العلاجات ذات النية الشافية قد تكون مناسبة حسب المرحلة.")

            elif intent == 'follow_up':
                if '0' in stage or 'dcis' in tumor:
                    if lang == 'en':
                        parts.append("For DCIS, surveillance with annual imaging is common; follow-up intervals should be per local guidelines and patient-specific factors.")
                    else:
                        parts.append("في حالات DCIS، المتابعة عادةً تكون بتصوير سنوي؛ تحدد فترات المتابعة حسب الإرشادات المحلية وعوامل المريضة.")
                else:
                    if lang == 'en':
                        parts.append("Follow-up frequency is individualized; early-stage disease often has clinic visits every 3-6 months in the first 2 years, then less frequent checks.")
                    else:
                        parts.append("تختلف وتيرة المتابعة بحسب المرحلة؛ غالبًا تكون الزيارات كل 3-6 أشهر في السنتين الأوليين للحالات المبكرة ثم تقل تدريجيًا.")

            elif intent == 'biomarker':
                if 'positive' in her2:
                    if lang == 'en':
                        parts.append("HER2-positive disease typically benefits from HER2-targeted therapy (e.g., trastuzumab-containing regimens); discuss specifics with medical oncology.")
                    else:
                        parts.append("الحالات HER2 موجبة تستفيد عادةً من علاجات موجهة لـ HER2 (مثل trastuzumab)؛ ناقش التفاصيل مع أخصائي الأورام.")
                if 'positive' in er or 'positive' in pr:
                    if lang == 'en':
                        parts.append("ER/PR-positive disease commonly includes endocrine therapy (e.g., tamoxifen or aromatase inhibitors) as part of management.")
                    else:
                        parts.append("الحالات ER/PR موجبة غالبًا تتضمن علاجًا هرمونيًا (مثل tamoxifen أو مثبطات الأروماتاز) كجزء من الخطة.")
                if not parts:
                    if lang == 'en':
                        parts.append("Please provide the biomarker details; management depends on exact ER/PR/HER2 status.")
                    else:
                        parts.append("يرجى توفير تفاصيل المؤشرات الحيوية؛ تعتمد الخطة على حالة ER/PR/HER2 الدقيقة.")

            elif intent == 'nutrition':
                # Basic nutrition guidance (approximate) — encourage dietitian consult and weight-based calculation
                # check for renal disease or other comorbidities that may alter protein needs
                renal = any('renal' in c or 'kidney' in c or 'ckd' in c for c in comorbidities)
                diabetic = any('diabetes' in c or 'diabetes' in c for c in comorbidities)
                # protein recommendations in g/kg/day
                if lang == 'en':
                    if renal:
                        parts.append("Protein needs should be individualized in patients with kidney disease; please consult a dietitian and nephrologist. Generally, typical cancer recommendations (1.2–1.5 g/kg/day) may need adjustment.")
                    else:
                        parts.append("Cancer patients commonly need more protein than average — roughly 1.2–1.5 g/kg/day; in highly catabolic states up to 1.5–2.0 g/kg/day. Provide patient weight for a precise calculation and refer to a clinical dietitian.")
                        if diabetic:
                            parts.append("Also consider total energy and carbohydrate planning for patients with diabetes; a dietitian can tailor the plan.")
                else:
                    if renal:
                        parts.append("ينبغي تخصيص احتياج البروتين للمصابين بأمراض الكلى؛ يُنصح باستشارة أخصائي تغذية وطبيب كلى. عمومًا، توصيات السرطان (1.2–1.5 غ/كغ/اليوم) قد تحتاج تعديلًا.")
                    else:
                        parts.append("عادةً يحتاج مرضى السرطان لبروتين أكثر من الطبيعي — حوالي 1.2–1.5 غ/كغ/اليوم؛ وفي حالات الهدم الشديد قد يصل إلى 1.5–2.0 غ/كغ/اليوم. زودنا بوزن المريضة لحساب أدق واحجزي استشارة أخصائي تغذية.")
                        if diabetic:
                            parts.append("كما يجب مراعاة إجمالي الطاقة والكربوهيدرات في المرضى المصابين بالسكر؛ يمكن لأخصائي التغذية تكييف الخطة.")
+
             else:  # general
                 if 'positive' in er:
                     if lang == 'en':
                         parts.append("ER-positive status suggests endocrine therapy is likely part of management.")
                     else:
                         parts.append("حالة ER موجبة تشير إلى أن العلاج الهرموني سيكون جزءًا مهمًا من الخطة.")
                 if 'positive' in her2:
                     if lang == 'en':
                         parts.append("HER2-positive cases typically require HER2-targeted therapy.")
                     else:
                         parts.append("حالات HER2 موجبة عادةً تتطلب علاجًا موجهًا لـ HER2.")
                 if not parts:
                     if lang == 'en':
                         parts.append("This is a complex clinical question; please review imaging, pathology, and multidisciplinary recommendations.")
                     else:
                         parts.append("هذا سؤال طبي معقَّد؛ يرجى مراجعة التصوير والأنسجة وتوصيات فريق متعدد التخصصات.")

            res = ' '.join(parts[:5])
            if lang == 'en':
                res += "\n\n(Note: General information only; not a substitute for medical advice.)"
            else:
                res += "\n\n(ملاحظة: هذه معلومات عامة وليست بديلاً عن استشارة الطبيب.)"
            return res

        # use intent-aware fallback when generator output is empty or appears to echo the prompt
        intent = _detect_intent(question)
        try:
            rule_ans = _rule_based_answer(p, question, intent=intent, lang=lang)
        except Exception as e:
            logger.error(f"Rule-based answer generation failed: {e}")
            rule_ans = ("عذرًا، حصلت مشكلة في توليد الإجابة الآن. " if lang=='ar' else "Sorry, something went wrong generating the answer.")

        # Heuristic to detect prompt echo / unusable generator output
        # Heuristic to detect prompt echo / unusable generator output
        def _looks_like_prompt_echo(ans: str, prompt_head: str) -> bool:
            if not ans or not ans.strip():
                return True
            s = ans.strip()
            if len(s) < 20:
                return True
            if prompt_head[:30] in s or s[:30] in prompt_head:
                return True
            if any(marker in s for marker in ['أنت مساعد طبي افتراضي', 'You are a virtual medical assistant', 'معلومات المريضة', 'Patient info', 'الاستعلام', 'Query']):
                return True
            return False

        looks_like_prompt = _looks_like_prompt_echo(answer, header)

        if not answer or looks_like_prompt:
            try:
                rule_ans = _rule_based_answer(p, question, lang=lang)
                answer = rule_ans
                used_fallback = True
            except Exception as e:
                logger.error(f"Rule-based answer generation failed: {e}")
                if not answer:
                    answer = ("عذرًا، حصلت مشكلة في توليد الإجابة الآن. " if lang=='ar' else "Sorry, there was a problem generating the answer right now.")

        else:
            if used_fallback:
                note = "\n\n(ملاحظة: تمت الإجابة باستخدام موفر محلي للتجربة)" if lang=='ar' else "\n\n(Note: answered using local provider for testing)"
                answer = f"{answer}{note}"

        return {"answer": answer, "sources": retrieved, "patient_id": patient_id}
