# tests/test_model.py
import io
import os
import json
import yaml
import faiss
import subprocess
from pathlib import Path
import sys

import pytest

# Proje köküne göre yollar
CFG_PATH = Path("configs/config.yaml")
META_PATH = None
CHUNKS_PATH = None
FAISS_PATH = None

@pytest.fixture(scope="session")
def cfg():
    assert CFG_PATH.exists(), "configs/config.yaml bulunamadı."
    with io.open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def ensure_index(cfg):
    """
    İndeks dosyaları yoksa önce oluşturur.
    """
    global META_PATH, CHUNKS_PATH, FAISS_PATH
    META_PATH = Path(cfg["indexing"]["meta_path"])
    CHUNKS_PATH = Path(cfg["indexing"]["chunks_path"])
    FAISS_PATH = Path(cfg["indexing"]["faiss_index_path"])

    if not (META_PATH.exists() and CHUNKS_PATH.exists() and FAISS_PATH.exists()):
        # İndeksi kur
        cmd = ["python", "main.py", "build-index", "--config", str(CFG_PATH)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"İndeks oluşturulamadı:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Son kontrol
    assert META_PATH.exists(), "index_meta.json yok."
    assert CHUNKS_PATH.exists(), "chunks.jsonl yok."
    assert FAISS_PATH.exists(), "index.faiss yok."

    return True

def test_config_minimal_keys(cfg):
    # Gerekli anahtarların varlığı
    for section in ["data", "indexing"]:
        assert section in cfg, f"config.yaml içinde '{section}' bölümü yok."
    for key in ["raw_dir", "processed_dir"]:
        assert key in cfg["data"], f"'data.{key}' yok."
    for key in ["chunk_size", "chunk_overlap", "embedding_model", "faiss_index_path", "meta_path", "chunks_path"]:
        assert key in cfg["indexing"], f"'indexing.{key}' yok."

def test_artifacts_and_meta(ensure_index, cfg):
    # Meta ile dosyalar uyumlu mu?
    with io.open(cfg["indexing"]["meta_path"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    # chunks satır sayısı == meta.size
    with io.open(cfg["indexing"]["chunks_path"], "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)
    assert num_lines == int(meta["size"]), "chunks.jsonl satır sayısı meta.size ile uyuşmuyor."

    # FAISS yüklenebiliyor mu ve ntotal == size mi?
    index = faiss.read_index(cfg["indexing"]["faiss_index_path"])
    assert index.ntotal == int(meta["size"]), "FAISS 'ntotal' ile meta.size eşleşmiyor."

def test_predictor_search_and_answer(ensure_index, cfg):
    # Basit bir arama ve cevap üretimi hatasız çalışmalı
    from src.models.predict_model import RAGPredictor

    rag = RAGPredictor(str(CFG_PATH))
    # Herhangi bir sorgu ile en azından 1 sonuç dönebilmeli (indeks boş değilse)
    results = rag.search("test", top_k=3)
    assert isinstance(results, list), "search list döndürmeli."
    if len(results) > 0:
        # Beklenen alanlar mevcut mu?
        for key in ["score", "id", "source", "text"]:
            assert key in results[0], f"Sonuç alanı eksik: {key}"

    # Cevap metni üretimi
    out = rag.answer("test", top_k=3)
    assert isinstance(out, str) and len(out) > 0, "answer boş çıktı üretti."

def test_cli_query_runs(ensure_index):
    # CLI ile sorgu akışı hatasız dönmeli (UTF-8 güvenli)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # çocuk süreç UTF-8 yazsın
    cmd = [sys.executable, "main.py", "query", "deneme"]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        encoding="utf-8",              # ana süreç UTF-8 decode etsin
    )
    assert res.returncode == 0, f"CLI 'query' komutu hata verdi:\nSTDERR:\n{res.stderr}"
    assert "Soru:" in res.stdout, "CLI çıktısı beklenen formatta değil (Soru: bulunamadı)."

