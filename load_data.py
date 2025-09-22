import os
import django
from dotenv import load_dotenv

# โหลด ENV (.env ต้องอยู่ใน D:\DWRag\)
load_dotenv()

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "DWRag.settings")
django.setup()

from core.models import MyDocument
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ ใช้ HF Token จาก .env
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("❌ HUGGING_FACE_HUB_API_TOKEN ไม่ถูกตั้งค่าใน .env")

# สร้าง embedder (ใช้ HuggingFace)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------- Data ตัวอย่าง ----------
docs = [
    {"text": "Python เป็นภาษาโปรแกรมที่ได้รับความนิยม", "source": "wiki"},
    {"text": "Django เป็น web framework ที่เขียนด้วย Python", "source": "wiki"},
    {"text": "Flutter ใช้พัฒนา mobile app แบบ cross-platform", "source": "wiki"},
]

# ---------- บันทึกลง Database ----------
for d in docs:
    embedding = embedder.embed_query(d["text"])
    MyDocument.objects.create(
        text=d["text"],
        source=d["source"],
        embedding=embedding
    )
    print(f"✅ เพิ่มข้อมูล: {d['text'][:30]}...")

print("🎉 โหลดข้อมูลเสร็จแล้ว")
