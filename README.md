Simulation experiments for two data market scenarios:
- Mean Estimation Market
- Data Mixture Market
- RAG Market

The following payment rules are compared:
- Leave-One-Out
- Data Shapley
- Myerson payment
- VCG payment

All the questions generated from `generate_questions.py` are stored in `data/ground-truth-retrieval.csv`.

The scores returned by LLM judge (DeepSeek R1 here) and the corresponding answers are stored in `data/query_w_llm_generated_ans_with_judge_scores.csv`
