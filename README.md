Simulation experiments for two data market scenarios:
- Mean Estimation Market
- Data Mixture Market
- RAG Market

The following payment rules are compared:
- Leave-One-Out
- Data Shapley
- Myerson payment
- VCG payment

## RAG Market

Retrieve Top 10 documents based on semantic similarity, and rerank the top10 documents based on the judge score and reported costs. To run the experiments:


1. First, generate the FAISS vector store using `data/medquad.csv`:

```bash
python3 get_embeddings.py
```

2. Set $INDEX_DIR to the faiss_index path, and run 
```bash
python3 RAG_reranking_with_cost_llm_utility.py
```
The results will be saved to `res.json`.