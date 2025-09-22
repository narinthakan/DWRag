import os, json, traceback
from functools import lru_cache

from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import F, Value, FloatField, ExpressionWrapper
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from .models import MyDocument
from pgvector.django import CosineDistance

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from django.views.decorators.csrf import csrf_exempt   # ✅ ปิด CSRF ชั่วคราว

# ---------- ENV ----------
load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HUGGING_FACE_HUB_API_TOKEN ไม่ถูกตั้งค่าใน .env")
os.environ["HUGGING_FACE_HUB_API_TOKEN"] = HF_TOKEN

# ---------- GLOBALS ----------
EMBEDDER = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",   # ✅ ใช้ flash (ไวกว่า pro)
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2,
        max_output_tokens=512,
    )

# ---------- PROMPTS ----------
RAG_PROMPT = """
คุณเป็นผู้ช่วยที่ให้คำตอบโดยอ้างอิงจาก Context ด้านล่างเท่านั้น
ถ้า Context ไม่มีคำตอบ ให้ตอบว่า: "ฉันไม่มีข้อมูลเพียงพอที่จะตอบคำถามนี้"

Context:
{context}

คำถาม:
{question}

โปรดตอบเป็นภาษาไทย 2–3 ประโยค
"""

OPEN_PROMPT = """
You are a helpful assistant. Explain clearly and concisely.

Question:
{question}

Answer in Thai, 2–3 sentences, give a short example if useful.
"""

GREETINGS = {"hi", "hello", "hey", "สวัสดี", "หวัดดี", "เฮลโล่"}

def _fast_greeting_reply(q: str) -> str | None:
    qn = (q or "").strip().lower()
    if qn in GREETINGS or any(qn.startswith(g) for g in GREETINGS):
        return "สวัสดีค่ะ 😊 ฉันช่วยตอบคำถามจากเอกสารของคุณได้ ลองถามว่า “Python คืออะไร” หรือ “มีหัวข้ออะไรในเอกสารบ้าง?”"
    return None

def _force_to_text(x) -> str:
    try:
        if x is None:
            return ""
        if hasattr(x, "content"):
            return str(x.content)
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            if "generated_text" in x:
                return str(x["generated_text"])
            if "text" in x:
                return str(x["text"])
            return json.dumps(x, ensure_ascii=False)
        if isinstance(x, list) and x:
            return _force_to_text(x[0])
        return str(x)
    except Exception:
        return ""

# ---------- MAIN VIEW ----------
@csrf_exempt   # ✅ ปิด CSRF (ใช้ตอน dev/test)
def rag_query_view(request):
    if request.method != "POST":
        return render(request, "core/rag_query.html")

    try:
        user_query = (request.POST.get("query") or "").strip()
        if not user_query and request.body:
            try:
                data = json.loads(request.body.decode("utf-8"))
                user_query = (data.get("query") or "").strip()
            except Exception:
                pass

        if not user_query:
            return JsonResponse({"answer": "กรุณาใส่คำถาม"}, status=400)

        # Quick reply
        quick = _fast_greeting_reply(user_query)
        if quick:
            return JsonResponse({"answer": quick})

        llm = get_llm()

        # คำถามสั้น ๆ → ส่งไป Open
        if len(user_query) <= 2:
            prompt = PromptTemplate.from_template(OPEN_PROMPT)
            chain = RunnableMap({"question": lambda _: user_query}) | prompt | llm
            raw = chain.invoke({})
            return JsonResponse({"answer": _force_to_text(raw).strip() or "ขออภัย ตอนนี้ยังไม่สามารถให้คำตอบได้"})

        # Embed + ค้นหา
        qvec = EMBEDDER.embed_query(user_query)
        qs = (
            MyDocument.objects
            .annotate(distance=CosineDistance("embedding", qvec))
            .annotate(similarity=ExpressionWrapper(Value(1.0) - F("distance"), output_field=FloatField()))
            .order_by("-similarity")
        )
        THRESHOLD = 0.28
        hits = list(qs.filter(similarity__gte=THRESHOLD)[:5])

        if hits:
            # --- RAG mode ---
            context = "\n".join(d.text for d in hits)
            best_sim = float(getattr(hits[0], "similarity", 0.0))
            print(f"🔎 MODE=RAG | hits={len(hits)} | best_sim={best_sim:.3f}")

            prompt = PromptTemplate.from_template(RAG_PROMPT)
            chain = RunnableMap({"context": lambda _: context, "question": lambda _: user_query}) | prompt | llm
            raw = chain.invoke({})
            answer = _force_to_text(raw).strip()
        else:
            # --- Open mode ---
            print(f"🔎 MODE=OPEN | no hits")
            prompt = PromptTemplate.from_template(OPEN_PROMPT)
            chain = RunnableMap({"question": lambda _: user_query}) | prompt | llm
            raw = chain.invoke({})
            answer = _force_to_text(raw).strip()

        if not answer:
            answer = "ขออภัย ตอนนี้ยังไม่สามารถให้คำตอบได้"

        return JsonResponse({"answer": answer})

    except Exception as e:
        print("❌ VIEW ERROR:", repr(e))
        traceback.print_exc()
        return JsonResponse({"answer": f"เกิดข้อผิดพลาด: {e}"}, status=500)


def home_view(request):
    return render(request, "core/rag_query.html")
