# src/models/predict_model.py
import os
import io
import json
import yaml
import faiss
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer

# Opsiyonel: openai mevcutsa kullanalım, yoksa ekstraktife düşelim
try:
    import openai  # pip install openai (opsiyonel)
except Exception:
    openai = None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    # id alanına göre sıralayalım; FAISS indeksi ekleme sırasını takip eder
    items.sort(key=lambda x: int(x["id"]))
    return items


class RAGPredictor:
    """
    Kolay Proje senaryosu:
    - FAISS indeksini ve chunk'ları diskten yükler
    - Sorguyu gömerek en benzer parçaları bulur
    - 'extractive' modda en alakalı pasajları derleyip yanıt döner
    - 'llm' modunda (opsiyonel) pasajları bağlama ekleyip LLM ile yanıtlar
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = self._load_cfg(config_path)
        self._load_meta_and_paths()
        self._load_index()
        self._load_chunks()
        self._load_embedder()

    # --------------------------- yükleyiciler ---------------------------------

    def _load_cfg(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_meta_and_paths(self) -> None:
        idx_cfg = self.cfg["indexing"]
        self.faiss_index_path = idx_cfg["faiss_index_path"]
        self.meta_path = idx_cfg["meta_path"]
        self.chunks_path = idx_cfg["chunks_path"]

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # Uyumluluk kontrolleri
        if self.meta["embedding_model"] != self.cfg["indexing"]["embedding_model"]:
            raise RuntimeError(
                "Konfig ile indeks uyumsuz: embedding_model farklı. "
                "İndeksi yeniden inşa edin veya config'i meta ile uyumlu yapın."
            )

        self.normalize = bool(self.meta.get("normalize_embeddings", True))
        self.index_type = self.meta.get("index_type", "IP")

    def _load_index(self) -> None:
        if not Path(self.faiss_index_path).exists():
            raise FileNotFoundError(
                f"FAISS indeks bulunamadı: {self.faiss_index_path}. "
                "Önce indeksleme adımını çalıştırın."
            )
        self.index = faiss.read_index(self.faiss_index_path)

    def _load_chunks(self) -> None:
        if not Path(self.chunks_path).exists():
            raise FileNotFoundError(
                f"Chunks JSONL bulunamadı: {self.chunks_path}. "
                "Önce indeksleme adımını çalıştırın."
            )
        self.chunks = _read_jsonl(self.chunks_path)

    def _load_embedder(self) -> None:
        model_name = self.cfg["indexing"]["embedding_model"]
        self.embedder = SentenceTransformer(model_name)
        # Boyut uyumluluğu (fail-fast)
        dim = self.embedder.get_sentence_embedding_dimension()
        if int(self.meta["dim"]) != int(dim):
            raise RuntimeError(
                f"Gömme boyutu uyumsuz: meta dim={self.meta['dim']} vs model dim={dim}. "
                "İndeksi yeniden inşa edin."
            )

    # ----------------------------- çekirdek API --------------------------------

    def embed_query(self, text: str) -> np.ndarray:
        v = self.embedder.encode(
            [text],
            normalize_embeddings=self.normalize,
        )
        return np.asarray(v, dtype=np.float32)  # (1, dim)

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = int(self.cfg.get("retrieval", {}).get("top_k", 5))

        q = self.embed_query(query)
        D, I = self.index.search(q, top_k)  # D: skor/mesafe, I: indeksler
        D, I = D[0], I[0]

        results: List[Dict[str, Any]] = []
        for score, idx in zip(D.tolist(), I.tolist()):
            if idx < 0:
                continue  # FAISS doldurma değeri
            chunk = self.chunks[idx]
            # IP ise büyük daha iyi; L2 ise küçük daha iyi → skoru normalize edelim
            if self.index_type.upper() == "L2":
                score = -float(score)  # daha yüksek = daha iyi olacak şekilde ters çevir
            results.append(
                {
                    "rank": len(results) + 1,
                    "score": float(score),
                    "id": chunk["id"],
                    "source": chunk["source"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                    "text": chunk["text"],
                }
            )
        return results

    # ---------------------------- cevap üretimi --------------------------------

    def _format_sources(self, hits: List[Dict[str, Any]]) -> str:
        # Aynı dosya tekrarlarını sadeleştir
        uniq = []
        seen = set()
        for h in hits:
            key = (h["source"], h["char_start"], h["char_end"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(h)
        lines = []
        for i, h in enumerate(uniq, 1):
            rng = f"{h['char_start']}-{h['char_end']}"
            lines.append(f"[{i}] {h['source']}#{rng}")
        return "\n".join(lines)

    def build_extractive_answer(self, query: str, hits: List[Dict[str, Any]], max_passages: int = 4) -> str:
        hits = hits[:max_passages]
        ctx = "\n\n---\n\n".join([h["text"] for h in hits])
        src = self._format_sources(hits)
        return (
            f"Soru: {query}\n\n"
            f"Cevap (ekstraktif, en alakalı pasajlardan derlenmiştir):\n"
            f"{ctx}\n\n"
            f"Kaynaklar:\n{src}"
        )

    def build_llm_answer(self, query: str, hits: List[Dict[str, Any]], max_passages: int = 4) -> str:
        if openai is None or not os.environ.get("OPENAI_API_KEY"):
            # Güvenli düşüş: ekstraktif
            return self.build_extractive_answer(query, hits, max_passages=max_passages)

        passages = hits[:max_passages]
        ctx_blocks = []
        for i, h in enumerate(passages, 1):
            ctx_blocks.append(f"[{i}] {h['text']}\n(Kaynak: {h['source']} #{h['char_start']}-{h['char_end']})")
        context = "\n\n".join(ctx_blocks)

        prompt = (
            "Aşağıdaki bağlam pasajlarını KESİNLİKLE kullanarak, kısa ve doğru bir Türkçe cevap yaz. "
            "Emin olmadığın kısımlar için 'bağlamda yok' de. Cevabın sonunda köşeli parantezle [1], [2] gibi "
            "kullandığın pasaj numaralarına atıf yap.\n\n"
            f"BAĞLAM:\n{context}\n\n"
            f"SORU: {query}"
        )

        # Eski ve yeni OpenAI SDK API'leri arasında uyumlu bir çağrı deneyelim
        model_name = self.cfg.get("answer", {}).get("openai_model", "gpt-4o-mini")
        try:
            # Eski tarz (ChatCompletion)
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Sen bağlamlı bir yardımcı asistansın."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            try:
                # Yeni tarz (Responses API benzeri adlar) varsa
                client = openai.OpenAI()  # type: ignore[attr-defined]
                resp = client.chat.completions.create(  # type: ignore[attr-defined]
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "Sen bağlamlı bir yardımcı asistansın."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()  # type: ignore[union-attr]
            except Exception:
                # Son çare: ekstraktif
                return self.build_extractive_answer(query, hits, max_passages=max_passages)

    def answer(self, query: str, top_k: int = None) -> str:
        hits = self.search(query, top_k=top_k)
        mode = self.cfg.get("answer", {}).get("mode", "extractive").lower()
        max_passages = int(self.cfg.get("answer", {}).get("max_passages", 4))
        if mode == "llm":
            return self.build_llm_answer(query, hits, max_passages=max_passages)
        else:
            return self.build_extractive_answer(query, hits, max_passages=max_passages)


# ------------------------------ CLI ------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kolay RAG: sorgu & cevap")
    parser.add_argument("question", type=str, help="Soru metni (tırnak içinde)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--k", type=int, default=None, help="Top-K override")
    args = parser.parse_args()

    predictor = RAGPredictor(args.config)
    out = predictor.answer(args.question, top_k=args.k)
    print(out)
