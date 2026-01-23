from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from helpers.config import Settings, get_settings

test_router = APIRouter(
    prefix="/test",
    tags=["Test"],
)

@test_router.get("/")
async def test_basic():
    """اختبار بسيط للتأكد أن الـ API يعمل"""
    return {
        "status": "success",
        "message": "API is working! 🚀",
        "test": True
    }

@test_router.get("/settings")
async def test_settings(app_settings: Settings = Depends(get_settings)):
    """اختبار إعدادات التطبيق"""
    return {
        "app_name": app_settings.APP_NAME,
        "app_version": app_settings.APP_VERSION,
        "generation_backend": app_settings.GENERATION_BACKEND,
        "embedding_backend": app_settings.EMBEDDING_BACKEND,
        "has_openai_key": bool(app_settings.OPENAI_API_KEY and app_settings.OPENAI_API_KEY.strip()),
        "has_cohere_key": bool(app_settings.COHERE_API_KEY and app_settings.COHERE_API_KEY.strip()),
        "has_gemini_key": bool(app_settings.GEMINI_API_KEY and app_settings.GEMINI_API_KEY.strip()),
    }

@test_router.post("/echo")
async def test_echo(message: dict):
    """يردد أي رسالة ترسلها له"""
    return {
        "status": "success",
        "your_message": message,
        "echo": f"You said: {message.get('text', 'nothing')}",
        "timestamp": "2026-01-03"
    }

@test_router.get("/hello/{name}")
async def test_hello(name: str):
    """رد شخصي بالاسم"""
    return {
        "status": "success",
        "message": f"Hello {name}! 👋",
        "greeting": f"مرحباً {name}، الـ API يعمل بنجاح!"
    }

@test_router.get("/gemini")
async def test_gemini_key(app_settings: Settings = Depends(get_settings)):
    """اختبار فعلي لـ Gemini API key"""
    
    if not app_settings.GEMINI_API_KEY:
        return {
            "status": "error",
            "message": "Gemini API key not found",
            "working": False
        }
    
    try:
        # استيراد Gemini
        from google import genai
        
        # إنشاء client
        client = genai.Client(api_key=app_settings.GEMINI_API_KEY)
        
        # اختبار بسيط - طريقة مبسطة
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello in Arabic"
        )
        
        return {
            "status": "success",
            "message": "Gemini API key is working! 🎉",
            "working": True,
            "gemini_response": response.text if hasattr(response, 'text') else str(response),
            "api_key_preview": app_settings.GEMINI_API_KEY[:10] + "..." if app_settings.GEMINI_API_KEY else None
        }
        
    except ImportError:
        return {
            "status": "error", 
            "message": "google-genai library not installed. Run: pip install google-genai",
            "working": False
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Gemini API key test failed: {str(e)}",
            "working": False,
            "error_type": type(e).__name__
        }

@test_router.get("/gemini/full")
async def test_gemini_full_performance(app_settings: Settings = Depends(get_settings)):
    """اختبار شامل لأداء Gemini في Generation و Embedding"""
    
    if not app_settings.GEMINI_API_KEY:
        return {
            "status": "error",
            "message": "Gemini API key not found",
            "working": False
        }
    
    results = {
        "generation": {},
        "embedding": {},
        "overall": {}
    }
    
    try:
        from google import genai
        import time
        
        # إنشاء client
        client = genai.Client(api_key=app_settings.GEMINI_API_KEY)
        
        # =========================
        # اختبار Text Generation
        # =========================
        try:
            start_time = time.time()
            
            generation_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="اكتب فقرة قصيرة عن أهمية الذكاء الاصطناعي في الطب"
            )
            
            generation_time = time.time() - start_time
            
            results["generation"] = {
                "status": "success",
                "working": True,
                "response_time_seconds": round(generation_time, 3),
                "model_used": "gemini-2.5-flash",
                "response_length": len(generation_response.text) if hasattr(generation_response, 'text') else 0,
                "sample_response": (generation_response.text[:200] + "...") if hasattr(generation_response, 'text') and len(generation_response.text) > 200 else getattr(generation_response, 'text', 'No text response'),
                "performance": "fast" if generation_time < 2 else "medium" if generation_time < 5 else "slow"
            }
            
        except Exception as e:
            results["generation"] = {
                "status": "error", 
                "working": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
        
        # =========================
        # اختبار Text Embedding
        # =========================
        try:
            start_time = time.time()
            
            # نصوص للاختبار
            test_texts = [
                "الذكاء الاصطناعي يساعد في تشخيص الأمراض",
                "Machine learning improves medical diagnosis", 
                "التكنولوجيا الطبية الحديثة"
            ]
            
            embedding_response = client.models.embed_content(
                model="text-embedding-004",
                contents=test_texts[0]
            )
            
            embedding_time = time.time() - start_time
            
            # تجربة embedding متعدد
            start_time_batch = time.time()
            batch_responses = []
            for text in test_texts:
                batch_response = client.models.embed_content(
                    model="text-embedding-004", 
                    contents=text
                )
                batch_responses.append(batch_response)
            
            batch_time = time.time() - start_time_batch
            
            results["embedding"] = {
                "status": "success",
                "working": True,
                "single_embedding": {
                    "response_time_seconds": round(embedding_time, 3),
                    "model_used": "text-embedding-004",
                    "embedding_dimension": len(embedding_response.embedding) if hasattr(embedding_response, 'embedding') else 0,
                    "performance": "fast" if embedding_time < 1 else "medium" if embedding_time < 3 else "slow"
                },
                "batch_embedding": {
                    "texts_count": len(test_texts),
                    "total_time_seconds": round(batch_time, 3),
                    "average_time_per_text": round(batch_time / len(test_texts), 3),
                    "performance": "fast" if batch_time < 3 else "medium" if batch_time < 8 else "slow"
                },
                "embedding_preview": embedding_response.embedding[:5] if hasattr(embedding_response, 'embedding') else "No embedding data"
            }
            
        except Exception as e:
            results["embedding"] = {
                "status": "error",
                "working": False, 
                "error": str(e),
                "error_type": type(e).__name__
            }
        
        # =========================
        # تقييم الأداء الإجمالي
        # =========================
        working_services = []
        if results["generation"].get("working", False):
            working_services.append("generation")
        if results["embedding"].get("working", False):
            working_services.append("embedding")
        
        overall_performance = "excellent"
        if results["generation"].get("working") and results["embedding"].get("working"):
            gen_perf = results["generation"].get("performance", "slow")
            emb_perf = results["embedding"]["single_embedding"].get("performance", "slow") 
            
            if "slow" in [gen_perf, emb_perf]:
                overall_performance = "good"
            elif "medium" in [gen_perf, emb_perf]:
                overall_performance = "very_good"
        elif len(working_services) == 1:
            overall_performance = "partial"
        else:
            overall_performance = "poor"
        
        results["overall"] = {
            "api_key_status": "working",
            "services_working": working_services,
            "services_count": f"{len(working_services)}/2",
            "overall_performance": overall_performance,
            "recommendation": "API ready for production" if len(working_services) == 2 else "Partial functionality - check errors"
        }
        
        return {
            "status": "success",
            "message": f"Gemini performance test completed! {len(working_services)}/2 services working",
            "api_key_preview": app_settings.GEMINI_API_KEY[:10] + "..." if app_settings.GEMINI_API_KEY else None,
            "test_results": results
        }
        
    except ImportError:
        return {
            "status": "error",
            "message": "google-genai library not installed. Run: pip install google-genai",
            "working": False
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Gemini full test failed: {str(e)}",
            "working": False,
            "error_type": type(e).__name__
        }

@test_router.post("/chat")
async def chat_with_gemini(message: dict, app_settings: Settings = Depends(get_settings)):
    """محادثة مع Gemini - شات بوت بسيط"""
    
    if not app_settings.GEMINI_API_KEY:
        return {
            "status": "error",
            "message": "Gemini API key not found",
            "response": "عذراً، الخدمة غير متوفرة الآن"
        }
    
    user_message = message.get("message", "").strip()
    if not user_message:
        return {
            "status": "error", 
            "message": "No message provided",
            "response": "من فضلك اكتب رسالة!"
        }
    
    try:
        from google import genai
        import time
        
        # إنشاء client
        client = genai.Client(api_key=app_settings.GEMINI_API_KEY)
        
        # تحسين الـ prompt للشات بوت
        enhanced_prompt = f"""
أنت مساعد ذكي ومفيد، تجيب بطريقة ودودة ومفيدة.
        
سؤال المستخدم: {user_message}

يرجى الإجابة بطريقة:
- واضحة ومفيدة
- مهذبة وودودة  
- مختصرة لكن شاملة
- باللغة العربية إذا كان السؤال بالعربية
"""
        
        start_time = time.time()
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=enhanced_prompt
        )
        
        response_time = time.time() - start_time
        
        bot_response = response.text if hasattr(response, 'text') else "عذراً، لم أتمكن من فهم طلبك"
        
        return {
            "status": "success",
            "user_message": user_message,
            "bot_response": bot_response,
            "response_time": round(response_time, 2),
            "model": "gemini-2.5-flash",
            "timestamp": "2026-01-03"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Chat failed: {str(e)}",
            "response": "عذراً، حدث خطأ في الخدمة",
            "user_message": user_message
        }

@test_router.post("/chat/medical")
async def medical_chat_with_gemini(message: dict, app_settings: Settings = Depends(get_settings)):
    """محادثة طبية متخصصة مع Gemini"""
    
    if not app_settings.GEMINI_API_KEY:
        return {
            "status": "error",
            "message": "Gemini API key not found",
            "response": "عذراً، الخدمة غير متوفرة الآن"
        }
    
    user_message = message.get("message", "").strip()
    if not user_message:
        return {
            "status": "error",
            "message": "No message provided", 
            "response": "من فضلك اكتب سؤالك الطبي!"
        }
    
    try:
        from google import genai
        import time
        
        client = genai.Client(api_key=app_settings.GEMINI_API_KEY)
        
        # prompt متخصص للاستشارات الطبية
        medical_prompt = f"""
أنت مساعد طبي ذكي متخصص في مجال الصحة وسرطان الثدي.

مهام:
- تقديم معلومات طبية دقيقة ومفيدة
- التوعية بأهمية الفحص المبكر
- تقديم الدعم النفسي والمعنوي
- التذكير بأهمية استشارة الطبيب المختص

تنبيه مهم: اذكر دائماً أن هذه معلومات عامة وليست بديلاً عن الاستشارة الطبية.

سؤال المريض: {user_message}

يرجى الإجابة بطريقة:
- علمية ودقيقة
- مطمئنة ومشجعة
- واضحة وسهلة الفهم
- مع التأكيد على أهمية استشارة الطبيب
"""
        
        start_time = time.time()
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=medical_prompt
        )
        
        response_time = time.time() - start_time
        
        bot_response = response.text if hasattr(response, 'text') else "عذراً، لم أتمكن من الإجابة على سؤالك الطبي"
        
        return {
            "status": "success",
            "user_message": user_message,
            "medical_response": bot_response,
            "response_time": round(response_time, 2),
            "model": "gemini-2.5-flash",
            "disclaimer": "هذه معلومات عامة وليست بديلاً عن الاستشارة الطبية المتخصصة",
            "timestamp": "2026-01-03"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Medical chat failed: {str(e)}",
            "response": "عذراً، حدث خطأ في الخدمة الطبية",
            "user_message": user_message
        }

@test_router.get("/all-apis")
async def test_all_api_keys(app_settings: Settings = Depends(get_settings)):
    """اختبار جميع مفاتيح الـ API دفعة واحدة"""
    results = {}
    
    # اختبار Gemini
    try:
        if app_settings.GEMINI_API_KEY:
            from google import genai
            client = genai.Client(api_key=app_settings.GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Say hello"
            )
            results["gemini"] = {"status": "success", "working": True, "response": "Hello from Gemini!"}
        else:
            results["gemini"] = {"status": "error", "working": False, "message": "API key not found"}
    except Exception as e:
        results["gemini"] = {"status": "error", "working": False, "message": str(e)}
    
    # اختبار OpenAI
    try:
        if app_settings.OPENAI_API_KEY:
            from openai import OpenAI
            client = OpenAI(api_key=app_settings.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=10
            )
            results["openai"] = {"status": "success", "working": True, "response": "Hello from OpenAI!"}
        else:
            results["openai"] = {"status": "error", "working": False, "message": "API key not found"}
    except Exception as e:
        results["openai"] = {"status": "error", "working": False, "message": str(e)}
    
    # اختبار Cohere
    try:
        if app_settings.COHERE_API_KEY:
            import cohere
            client = cohere.Client(api_key=app_settings.COHERE_API_KEY)
            response = client.generate(prompt="Say hello", max_tokens=10)
            results["cohere"] = {"status": "success", "working": True, "response": "Hello from Cohere!"}
        else:
            results["cohere"] = {"status": "error", "working": False, "message": "API key not found"}
    except Exception as e:
        results["cohere"] = {"status": "error", "working": False, "message": str(e)}
    
    # إحصائيات إجمالية
    working_count = sum(1 for result in results.values() if result.get("working", False))
    total_count = len(results)
    
    return {
        "status": "success",
        "message": f"API test completed: {working_count}/{total_count} APIs working",
        "summary": {
            "total_apis": total_count,
            "working_apis": working_count,
            "success_rate": f"{(working_count/total_count*100):.1f}%"
        },
        "details": results
    }

@test_router.get("/openai")
async def test_openai_key(app_settings: Settings = Depends(get_settings)):
    """اختبار فعلي لـ OpenAI API key"""
    
    if not app_settings.OPENAI_API_KEY:
        return {
            "status": "error",
            "message": "OpenAI API key not found",
            "working": False
        }
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=app_settings.OPENAI_API_KEY)
        
        # اختبار بسيط
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello in Arabic"}],
            max_tokens=50
        )
        
        return {
            "status": "success", 
            "message": "OpenAI API key is working! 🎉",
            "working": True,
            "openai_response": response.choices[0].message.content,
            "api_key_preview": app_settings.OPENAI_API_KEY[:10] + "..." if app_settings.OPENAI_API_KEY else None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"OpenAI API key test failed: {str(e)}",
            "working": False,
            "error_type": type(e).__name__
        }

@test_router.get("/cohere") 
async def test_cohere_key(app_settings: Settings = Depends(get_settings)):
    """اختبار فعلي لـ Cohere API key"""
    
    if not app_settings.COHERE_API_KEY:
        return {
            "status": "error",
            "message": "Cohere API key not found", 
            "working": False
        }
    
    try:
        import cohere
        client = cohere.Client(api_key=app_settings.COHERE_API_KEY)
        
        # اختبار بسيط
        response = client.generate(
            prompt="Say hello in Arabic",
            max_tokens=50
        )
        
        return {
            "status": "success",
            "message": "Cohere API key is working! 🎉", 
            "working": True,
            "cohere_response": response.generations[0].text.strip(),
            "api_key_preview": app_settings.COHERE_API_KEY[:10] + "..." if app_settings.COHERE_API_KEY else None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Cohere API key test failed: {str(e)}",
            "working": False,
            "error_type": type(e).__name__
        }
