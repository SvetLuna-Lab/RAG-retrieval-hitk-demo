import json
import os
from typing import Any, Dict, List

from retriever import SimpleBM25Retriever
from retrieval_metrics import RetrievalMetrics, evaluate_single_query


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
EVAL_QUERIES_PATH = os.path.join(DATA_DIR, "eval_queries.json")


def load_eval_queries(path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation queries from JSON.
    Expected fields: id, query, relevant_docs.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(top_k: int = 3) -> None:
    queries = load_eval_queries(EVAL_QUERIES_PATH)
    retriever = SimpleBM25Retriever(corpus_dir=CORPUS_DIR)

    results: List[Dict[str, Any]] = []
    metrics_list: List[RetrievalMetrics] = []

    for q in queries:
        qid = q["id"]
        query = q["query"]
        relevant_docs = q.get("relevant_docs", [])

        retrieved = retriever.retrieve(query, top_k=top_k)
        retrieved_ids = [doc.doc_id for doc, _ in retrieved]

        m = evaluate_single_query(retrieved_ids, relevant_docs)
        metrics_list.append(m)

        results.append(
            {
                "id": qid,
                "query": query,
                "relevant_docs": relevant_docs,
                "retrieved_docs": [
                    {"doc_id": doc.doc_id, "score": score} for doc, score in retrieved
                ],
                "metrics": {
                    "hit_at_k": m.hit_at_k,
                    "recall_at_k": m.recall_at_k,
                    "mrr": m.mrr,
                },
            }
        )

    # Aggregate simple averages
    n = len(metrics_list) or 1
    avg_hit = sum(m.hit_at_k for m in metrics_list) / n
    avg_recall = sum(m.recall_at_k for m in metrics_list) / n
    avg_mrr = sum(m.mrr for m in metrics_list) / n

    print("=== RAG retrieval evaluation summary ===")
    print(f"top_k = {top_k}")
    print(f"avg hit@k   = {avg_hit:.3f}")
    print(f"avg recall@k= {avg_recall:.3f}")
    print(f"avg MRR     = {avg_mrr:.3f}")

    out_path = os.path.join(PROJECT_ROOT, "retrieval_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
