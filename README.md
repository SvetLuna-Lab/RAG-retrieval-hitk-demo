# RAG-retrieval-hitk-demo

Small, self-contained demo of **RAG retrieval evaluation**.

The goal of this repository is to show, in a minimal way, how to:

- run a simple BM25-style retriever over a tiny corpus,
- define an evaluation set with **relevant documents** per query,
- compute basic **retrieval metrics**: Hit@k, Recall@k, and MRR.

The project focuses only on the **retrieval** part (no LLM, no answer generation).

---

## Requirements

- Python 3.10+  
No external dependencies are required (standard library only).


## Repository structure

```text
rag-retrieval-hitk-demo/
├─ data/
│  ├─ corpus/                # tiny technical corpus (.txt)
│  │  ├─ doc1.txt
│  │  ├─ doc2.txt
│  │  └─ doc3.txt
│  └─ eval_queries.json      # queries + relevant_docs
├─ src/
│  ├─ __init__.py
│  ├─ retriever.py           # SimpleBM25Retriever (bag-of-words BM25-like)
│  ├─ retrieval_metrics.py   # hit@k, recall@k, mrr
│  └─ run_retrieval_eval.py  # main script: run retrieval eval over all queries
├─ tests/
│  ├─ __init__.py
│  └─ test_retrieval_metrics.py  # unit tests for retrieval metrics
├─ README.md
├─ requirements.txt
└─ .gitignore



Data format
Corpus

data/corpus/ contains a small set of .txt documents:

doc1.txt

doc2.txt

doc3.txt

They describe basic ideas around RAG, evaluation, and harness design.



Evaluation queries

data/eval_queries.json is a list of query objects.
Example:

[
  {
    "id": "q1",
    "query": "components of a RAG pipeline",
    "relevant_docs": ["doc1.txt"]
  }
]



Fields:

id – query identifier.

query – natural language query string.

relevant_docs – list of corpus document filenames that should be considered relevant for this query.

This is enough to demonstrate standard retrieval metrics.


Retriever

Implemented in src/retriever.py as SimpleBM25Retriever.

Loads all .txt documents from data/corpus/.

Applies a simple tokenization: lowercase → split on whitespace → strip non-alphanumeric characters.

Computes BM25-like scores over a bag-of-words representation.

Returns the top-k documents and their scores for each query.

This retriever is intentionally simple and dependency-free, suitable for small demos and unit tests.



Retrieval metrics

Implemented in src/retrieval_metrics.py.

For a single query, given:

retrieved: ordered list of retrieved document IDs (top-k),

relevant_docs: list of relevant document IDs,

the module computes:

Hit@k

hit@k = 1.0 if any relevant doc is in retrieved, else 0.0



Recall@k

recall@k = (number of relevant docs in retrieved) / (total number of relevant docs)



MRR (Mean Reciprocal Rank)

MRR = 1 / rank of the first relevant doc in retrieved, or 0.0 if none is found

All three metrics for a single query are returned as a RetrievalMetrics dataclass.



Main evaluation script

src/run_retrieval_eval.py is the main entry point.

It:

Loads the evaluation queries from data/eval_queries.json.

Instantiates SimpleBM25Retriever over data/corpus/.

For each query:

retrieves the top-k documents,

computes Hit@k, Recall@k, and MRR.

Aggregates average metrics across all queries.

Prints a short summary to stdout.

Saves a detailed JSON report to retrieval_eval_results.json in the project root.

Example summary output:

=== RAG retrieval evaluation summary ===
top_k = 3
avg hit@k   = 1.000
avg recall@k= 0.833
avg MRR     = 0.889



Quick start

From the project root:

python src/run_retrieval_eval.py



This will:

Run the BM25-style retriever on all queries in data/eval_queries.json.

Compute Hit@k, Recall@k, and MRR per query.

Print average metrics.

Save detailed per-query results to:


retrieval_eval_results.json



You can inspect this JSON file to see, for each query:

the query text,

the list of relevant documents,

the retrieved documents with scores,

the metrics for that query.



Tests

The project includes a minimal test suite for the retrieval metrics.

Run from the project root:

python -m unittest discover -s tests



This will execute:

tests/test_retrieval_metrics.py – unit tests for hit_at_k, recall_at_k, and mrr_at_k.



Extending the demo

Possible next steps:

Add more queries and documents to stress-test the retriever.

Experiment with different values of top_k.

Replace SimpleBM25Retriever with a dense retriever using embeddings and vector search.

Integrate this retrieval evaluation with a RAG answer evaluation harness (e.g. rag-answer-eval-demo).



