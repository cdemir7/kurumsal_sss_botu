# === Base Image ===============================================================
FROM python:3.11-slim

# === Env & UTF-8 ============================================================== 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8

WORKDIR /app

# (İsteğe bağlı) Bazı hafif yardımcılar
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# === Python Bağımlılıkları ====================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === Proje Kaynakları =========================================================
COPY . .

# === Gömme Modelini Önceden İndir (soğuk başlatmayı azaltır) ==================
# Config ile uyumlu varsayılan: all-MiniLM-L6-v2
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
import os
model_name = os.environ.get('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2')
SentenceTransformer(model_name)  # indirilir ve cache'e alınır
print("[OK] Pre-fetched embedding model:", model_name)
PY

# === Varsayılan komut (yardım) ================================================
CMD ["python", "main.py", "--help"]
