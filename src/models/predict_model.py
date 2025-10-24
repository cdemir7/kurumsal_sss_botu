# src/models/predict_model.py
import os
import io
import json
import yaml
import faiss
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer

try:
    import openai  # opsiyonel
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
    items.sort(key=lambda x: int(x["id"]))
    return items


class RAGPredictor:
    """
    Kolay Proje (SSS/FAQ):
    - FAISS ve chunk'ları yükler
    - Sorguyu gömüp en yakın pasajları bulur
    - 'extractive' modda sadece ilgili cevabı kısa döndürür:
        1) FAQ kalıbı (S:/C:) varsa doğrudan Cevap bölümünü çıkar
        2) Yoksa en ilgili 1-2 cümleyi seçip döndür
      (Uzun pasajları asla tamamını yazmaz.)
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
        dim = self.embedder.get_sentence_embedding_dimension()
        if int(self.meta["dim"]) != int(dim):
            raise RuntimeError(
                f"Gömme boyutu uyumsuz: meta dim={self.meta['dim']} vs model dim={dim}. "
                "İndeksi yeniden inşa edin."
            )

    # ----------------------------- yardımcılar --------------------------------

    def embed_query(self, text: str) -> np.ndarray:
        v = self.embedder.encode([text], normalize_embeddings=self.normalize)
        return np.asarray(v, dtype=np.float32)  # (1, dim)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        embs = self.embedder.encode(texts, normalize_embeddings=self.normalize)
        return np.asarray(embs, dtype=np.float32)

    def _split_sentences(self, text: str) -> List[str]:
        # Basit TR dostu cümle bölücü: nokta, ?, ! ve satır sonları
        # Ayrıca Markdown başlık/boş satırlarını da kesme noktası kabul eder.
        raw = []
        for block in text.split("\n"):
            block = block.strip()
            if not block:
                continue
            raw.append(block)
        text = " ".join(raw)
        parts = []
        start = 0
        for i, ch in enumerate(text):
            if ch in ".!?":
                seg = text[start : i + 1].strip()
                if seg:
                    parts.append(seg)
                start = i + 1
        tail = text[start:].strip()
        if tail:
            parts.append(tail)
        return [s for s in parts if len(s) > 0]

    def _faq_pairs_from_text(self, text: str) -> List[Tuple[str, str]]:
        """
        'S:'/'Soru:' ve hemen sonrasındaki 'C:'/'Cevap:' satırlarını yakalar.
        Çok satırlı cevapları, bir sonraki S:/Soru: başlayana kadar biriktirir.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        pairs: List[Tuple[str, str]] = []
        q_buf: Optional[str] = None
        a_buf: List[str] = []

        def is_q(ln: str) -> Optional[str]:
            for pref in ("S:", "Soru:"):
                if ln.lower().startswith(pref.lower()):
                    return ln[len(pref):].strip()
            return None

        def is_a(ln: str) -> Optional[str]:
            for pref in ("C:", "Cevap:"):
                if ln.lower().startswith(pref.lower()):
                    return ln[len(pref):].strip()
            return None

        i = 0
        while i < len(lines):
            ln = lines[i]
            q = is_q(ln)
            if q is not None:
                # yeni bir soru başlıyor, önceki pair'i kapat
                if q_buf is not None and a_buf:
                    pairs.append((q_buf, " ".join(a_buf).strip()))
                    a_buf = []
                q_buf = q
                i += 1
                continue
            a = is_a(ln)
            if a is not None:
                # cevabı topla, bir sonraki Soru'ya kadar
                a_buf = [a]
                j = i + 1
                while j < len(lines) and is_q(lines[j]) is None:
                    nxt_a = is_a(lines[j])
                    if nxt_a is not None:
                        a_buf.append(nxt_a)
                    else:
                        a_buf.append(lines[j])
                    j += 1
                i = j
                continue
            i += 1

        if q_buf is not None and a_buf:
            pairs.append((q_buf, " ".join(a_buf).strip()))
        return pairs

    def _mmr_rerank(self, query_emb: np.ndarray, results: List[Dict[str, Any]], k: int, lambd: float) -> List[Dict[str, Any]]:
        """
        Basit MMR: lambd * sim(q, d) - (1 - lambd) * max_sim(d, S)
        Yalnızca FAISS'in döndürdüğü ilk k aday üzerinde çalışır.
        """
        if len(results) <= 1 or k <= 1:
            return results

        cand_texts = [r["text"] for r in results]
        cand_embs = self._embed_texts(cand_texts)  # (n, dim)
        q = query_emb.reshape(1, -1)                # (1, dim)

        selected: List[int] = []
        remaining = list(range(len(results)))

        # sim(q, d)
        q_sims = (cand_embs @ q.T).reshape(-1)  # cosine/IP varsayımı

        # ilk en iyi adayı seç
        best_first = int(np.argmax(q_sims))
        selected.append(best_first)
        remaining.remove(best_first)

        while len(selected) < min(k, len(results)) and remaining:
            mmr_scores = []
            for idx in remaining:
                # max sim(d, S)
                if selected:
                    sims_to_S = cand_embs[idx] @ cand_embs[selected].T  # (len(S),)
                    max_sim = float(np.max(sims_to_S))
                else:
                    max_sim = 0.0
                score = lambd * float(q_sims[idx]) - (1.0 - lambd) * max_sim
                mmr_scores.append((score, idx))
            mmr_scores.sort(reverse=True, key=lambda x: x[0])
            chosen = mmr_scores[0][1]
            selected.append(chosen)
            remaining.remove(chosen)

        return [results[i] for i in selected] + [results[i] for i in remaining if i not in selected]

    # ----------------------------- arama/cevap --------------------------------

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = int(self.cfg.get("retrieval", {}).get("top_k", 5))

        q = self.embed_query(query)
        D, I = self.index.search(q, top_k)
        D, I = D[0], I[0]

        results: List[Dict[str, Any]] = []
        for score, idx in zip(D.tolist(), I.tolist()):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if self.index_type.upper() == "L2":
                score = -float(score)  # yüksek = iyi
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

        # MMR (opsiyonel)
        use_mmr = bool(self.cfg.get("retrieval", {}).get("use_mmr", False))
        if use_mmr and results:
            lambd = float(self.cfg.get("retrieval", {}).get("mmr_lambda", 0.5))
            results = self._mmr_rerank(q[0], results, k=min(top_k, len(results)), lambd=lambd)

        return results

    def _format_sources(self, hits: List[Dict[str, Any]]) -> str:
        uniq = []
        seen = set()
        for h in hits:
            key = (h["source"], h["char_start"], h["char_end"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(h)
        lines = []
        for i, h in enumerate(uniq[:3], 1):  # en fazla 3 atıf göster
            rng = f"{h['char_start']}-{h['char_end']}"
            lines.append(f"[{i}] {h['source']}#{rng}")
        return "\n".join(lines)

    def build_extractive_answer(self, query: str, hits: List[Dict[str, Any]]) -> str:
        # 0) Skor eşiği kontrolü
        min_score = float(self.cfg.get("retrieval", {}).get("min_score", 0.0))
        if not hits or (hits and float(hits[0]["score"]) < min_score):
            return (
                f"Soru: {query}\n\n"
                "Cevap: Bağlamda güvenilir bir yanıt bulunamadı. "
                "Lütfen soruyu yeniden ifade edin ya da veri kaynağını genişletin."
            )

        max_passages = int(self.cfg.get("answer", {}).get("max_passages", 4))
        strategy = str(self.cfg.get("answer", {}).get("extractive_strategy", "faq_or_sentences")).lower()
        hits = hits[:max_passages]

        # 1) Önce FAQ kalıbını dene
        if "faq" in strategy:
            best_pair = None  # (score, answer, hit)
            q_emb = self.embed_query(query)[0]
            for h in hits:
                pairs = self._faq_pairs_from_text(h["text"])
                for (q_text, a_text) in pairs:
                    sim = float((self._embed_texts([q_text])[0] @ q_emb.T))
                    if (best_pair is None) or (sim > best_pair[0]):
                        best_pair = (sim, a_text.strip(), h)
            if best_pair is not None and best_pair[0] >= min_score:
                src = self._format_sources([best_pair[2]])
                return f"Soru: {query}\n\nCevap: {best_pair[1]}\n\nKaynaklar:\n{src}"

        # 2) Sonra cümle tabanlı ekstraksiyon
        max_sentences = int(self.cfg.get("answer", {}).get("max_sentences", 2))
        cand_sentences: List[Tuple[float, str, Dict[str, Any]]] = []
        q_emb = self.embed_query(query)[0]
        for h in hits:
            sents = self._split_sentences(h["text"])
            if not sents:
                continue
            sent_embs = self._embed_texts(sents)  # (m, dim)
            sims = (sent_embs @ q_emb.T).reshape(-1).tolist()
            for s, sc in zip(sents, sims):
                cand_sentences.append((float(sc), s.strip(), h))

        if not cand_sentences:
            src = self._format_sources(hits)
            return (
                f"Soru: {query}\n\n"
                "Cevap: Bağlamda açık bir yanıt çıkarılamadı.\n\n"
                f"Kaynaklar:\n{src}"
            )

        cand_sentences.sort(reverse=True, key=lambda x: x[0])
        chosen = cand_sentences[:max_sentences]
        answer_text = " ".join([c[1] for c in chosen]).strip()
        # Tekrarlayan aynı kaynaktan cümleleri tek atıfa indirgemek için benzersiz kaynakları topla
        chosen_sources = []
        seen = set()
        for _, _, h in chosen:
            key = (h["source"], h["char_start"], h["char_end"])
            if key not in seen:
                seen.add(key)
                chosen_sources.append(h)
        src = self._format_sources(chosen_sources)

        return f"Soru: {query}\n\nCevap: {answer_text}\n\nKaynaklar:\n{src}"

    # ---------------------------- LLM modu (opsiyonel) ------------------------

    def build_llm_answer(self, query: str, hits: List[Dict[str, Any]]) -> str:
        # Kısa ve atıflı LLM cevabı (opsiyonel). API yoksa ekstraktife düşer.
        if openai is None or not os.environ.get("OPENAI_API_KEY"):
            return self.build_extractive_answer(query, hits)

        max_passages = int(self.cfg.get("answer", {}).get("max_passages", 4))
        passages = hits[:max_passages]
        ctx_blocks = []
        for i, h in enumerate(passages, 1):
            ctx_blocks.append(f"[{i}] {h['text']}\n(Kaynak: {h['source']} #{h['char_start']}-{h['char_end']})")
        context = "\n\n".join(ctx_blocks)

        prompt = (
            "Aşağıdaki bağlam pasajlarını KESİNLİKLE kullanarak, kısa ve doğru bir Türkçe cevap yaz. "
            "Emin olmadığın kısımlar için 'bağlamda yok' de. Cevabın sonunda köşeli parantezle [1], [2] gibi "
            "kullandığın pasaj numaralarına atıf yap. Cevap 1-2 cümleyi geçmesin.\n\n"
            f"BAĞLAM:\n{context}\n\n"
            f"SORU: {query}"
        )

        model_name = self.cfg.get("answer", {}).get("openai_model", "gpt-4o-mini")
        try:
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
                return self.build_extractive_answer(query, hits)

    def answer(self, query: str, top_k: int = None) -> str:
        hits = self.search(query, top_k=top_k)
        mode = self.cfg.get("answer", {}).get("mode", "extractive").lower()
        if mode == "llm":
            return self.build_llm_answer(query, hits)
        else:
            return self.build_extractive_answer(query, hits)


# ------------------------------ CLI ------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kolay RAG: sorgu & cevap (kısa ve atıflı)")
    parser.add_argument("question", type=str, help="Soru metni (tırnak içinde)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--k", type=int, default=None, help="Top-K override")
    args = parser.parse_args()

    predictor = RAGPredictor(args.config)
    out = predictor.answer(args.question, top_k=args.k)
    print(out)
