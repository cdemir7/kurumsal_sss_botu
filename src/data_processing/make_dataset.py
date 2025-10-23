# src/data_processing/make_dataset.py
import os
import io
import json
import yaml
import faiss
import math
import glob
import regex as re
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

# --- Yardımcılar -------------------------------------------------------------

def read_text_file(path: str) -> str:
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def iter_raw_documents(raw_dir: str, exts=(".md", ".txt")) -> List[Tuple[str, str]]:
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(raw_dir, f"**/*{ext}"), recursive=True))
    docs = []
    for p in paths:
        try:
            txt = read_text_file(p)
            if txt and txt.strip():
                docs.append((p, txt))
        except Exception as e:
            print(f"[WARN] {p} okunamadı: {e}")
    return docs

def normalize_newlines(text: str) -> str:
    # Markdown/TXT için temel temizlik
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Fazla boş satırları sadeleştir
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def smart_chunk(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    """
    Basit ama pratik bir parçalayıcı:
    - Karakter tabanlı pencere
    - Kelime ortasında bölmemek için geriye doğru yakın boşluk/sonlandırıcı arar
    - start/end karakter indekslerini döndürür
    """
    n = len(text)
    if n == 0:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        # Kelime bölünmesini azalt: end civarında bir ayraç ara
        if end < n:
            window = text[start:end]
            # Tercihen paragraf sonu → satır sonu → boşluk
            candidates = [m.start() for m in re.finditer(r"\n\n|\n| ", window)]
            if candidates:
                # chunk_size'ın %80'inden önce kesmemeye çalış
                lower_bound = math.floor(0.8 * chunk_size)
                good = [c for c in candidates if c >= lower_bound]
                cut = (good[-1] if good else candidates[-1]) + start
                end = cut + 1  # ayıracı dâhil et
        segment = text[start:end].strip()
        if segment:
            chunks.append((start, end, segment))
        if end == n:
            break
        start = max(end - overlap, 0)

    return chunks

def build_faiss_index(embeddings: np.ndarray, use_ip: bool = True):
    """
    normalize_embeddings True ise kozmik açı (cosine) ≈ iç-çarpım ile çalışırız.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d) if use_ip else faiss.IndexFlatL2(d)
    index.add(embeddings.astype(np.float32))
    return index

# --- Ana İş Akışı ------------------------------------------------------------

def run_build_index(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    chunk_size = int(cfg["indexing"]["chunk_size"])
    chunk_overlap = int(cfg["indexing"]["chunk_overlap"])
    emb_model_name = cfg["indexing"]["embedding_model"]
    normalize_embeddings = bool(cfg["indexing"]["normalize_embeddings"])
    faiss_index_path = cfg["indexing"]["faiss_index_path"]
    meta_path = cfg["indexing"]["meta_path"]
    chunks_path = cfg["indexing"]["chunks_path"]

    print(f"[INFO] Kaynak klasör: {raw_dir}")
    docs = iter_raw_documents(raw_dir, exts=(".md", ".txt"))
    if not docs:
        raise SystemExit(
            f"[ERROR] '{raw_dir}' altında .md / .txt bulunamadı. "
            "Lütfen birkaç dosyayı data/raw/ altına koyup tekrar deneyin."
        )

    # Metni normalize et ve parçalara ayır
    all_chunks: List[Dict] = []
    for src_path, text in docs:
        text = normalize_newlines(text)
        pieces = smart_chunk(text, chunk_size=chunk_size, overlap=chunk_overlap)
        for i, (s, e, seg) in enumerate(pieces):
            all_chunks.append(
                {
                    "id": f"{len(all_chunks)}",
                    "source": str(Path(src_path).as_posix()),
                    "char_start": int(s),
                    "char_end": int(e),
                    "text": seg,
                }
            )

    if not all_chunks:
        raise SystemExit("[ERROR] Parçalama sonucu boş. Chunk parametrelerini kontrol edin.")

    print(f"[INFO] Toplam parça sayısı: {len(all_chunks)}")

    # Gömme modeli
    print(f"[INFO] Embedding modeli yükleniyor: {emb_model_name}")
    model = SentenceTransformer(emb_model_name)

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # FAISS indeks
    use_ip = True if normalize_embeddings else False  # normalize edildiyse IP iyi bir seçim
    index = build_faiss_index(embeddings, use_ip=use_ip)

    # Çıktıları yaz
    print(f"[INFO] FAISS indeks yazılıyor → {faiss_index_path}")
    faiss.write_index(index, faiss_index_path)

    print(f"[INFO] Meta yazılıyor → {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": emb_model_name,
                "normalize_embeddings": normalize_embeddings,
                "index_type": "IP" if use_ip else "L2",
                "size": len(all_chunks),
                "dim": int(embeddings.shape[1]),
                "faiss_index_path": faiss_index_path,
                "chunks_path": chunks_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[INFO] Parçalar yazılıyor → {chunks_path}")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("[OK] İndeksleme tamamlandı.")

# --- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kolay RAG: veri alımı ve indeksleme")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="YAML konfigürasyon yolu",
    )
    args = parser.parse_args()
    run_build_index(args.config)
