# main.py
import argparse
from src.data_processing.make_dataset import run_build_index
from src.models.predict_model import RAGPredictor

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
        run_build_index(args.config)
    elif args.cmd == "query":
        rag = RAGPredictor(args.config)
        print(rag.answer(args.question, top_k=args.k))

if __name__ == "__main__":
    main()
