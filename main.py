# main.py
import argparse
import sys, io

def _ensure_utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="Kolay RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="Ham veriden FAISS indeksi üret")
    b.add_argument("--config", default="configs/config.yaml")

    q = sub.add_parser("query", help="Sorgu çalıştır")
    q.add_argument("question", type=str)
    q.add_argument("--config", default="configs/config.yaml")
    q.add_argument("--k", type=int, default=None)

    args = parser.parse_args()

    if args.cmd == "build-index":
        from src.data_processing.make_dataset import run_build_index
        run_build_index(args.config)

    elif args.cmd == "query":
        from src.models.predict_model import RAGPredictor
        _ensure_utf8_stdout()  # <-- eklendi
        rag = RAGPredictor(args.config)
        print(rag.answer(args.question, top_k=args.k))

if __name__ == "__main__":
    main()
