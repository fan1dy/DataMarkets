import os
import sys
from dotenv import load_dotenv
from langchain.docstore.document import Document
from typing import List, Any
from langchain.chains import RetrievalQA
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
# from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field
from langchain.vectorstores import FAISS
import argparse
from datasets import load_dataset
import random
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

load_dotenv()

# Helper Classes and Functions

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")


def rerank_documents(docs: List[Document], scores: list, top_n: int = 1) -> List[Document]:
    prices = [doc.metadata["price"] for doc in docs]
    utilities =  [scores[i]-prices[i] for i in range(len(scores))]
    ranked_docs = list(zip(docs, utilities))
    # Sort by utility score in descending order
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    # Get Topn
    topn_docs = ranked_docs[:top_n]
    return topn_docs

class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        initial_res = self.vectorstore.similarity_search_with_relevance_scores(query, k = 10)
        initial_docs = [doc for doc, _ in initial_res]
        similarity_scores = [score for _, score in initial_res]
        print(similarity_scores)
        return rerank_documents(initial_docs, similarity_scores,top_n=num_docs)


def rag_run(query: str, docs: List[Document]) -> None:
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create FAISS vectorstore from documents
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    custom_retriever = CustomRetriever(vectorstore=vectorstore)
    advanced_docs = custom_retriever._get_relevant_documents(query, num_docs=1)
    return advanced_docs

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--query", type=str, default='What are the impacts of climate change?', help="Query to ask")
    parser.add_argument("--retriever_type", type=str, default="reranker", choices=["reranker", "cross_encoder"],
                        help="Type of retriever to use")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
    data = data.to_pandas()
    data["id"]=data.index
    MAX_ROWS = 1000
    DOCUMENT="Answer"
    TOPIC="qtype"
    subset_data = data.head(MAX_ROWS)
    doc_lists = subset_data["Answer"].tolist()
    price_list = [random.uniform(0, 0.05) for _ in range(MAX_ROWS)]
    docs = [Document(page_content=doc_lists[i], metadata={"price": price_list[i]}) for i in range(len(doc_lists))]

    
    import torch
    from torch import cuda
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    question_def = "What is a Sydenham chorea?"
    results = rag_run(query=question_def, docs=docs)
    print(results)

    # from transformers import AutoTokenizer, AutoModelForCausalLM

    # model_id = "Qwen/Qwen2.5-3B"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id,
    #                                             device_map=device,
    #                                             torch_dtype=torch.bfloat16)
    # question_def = "Write in 20 words what is a Sydenham chorea."
    # results = rag_run(query="Write in 20 words what is a Sydenham chorea.", docs=docs)
    # prompt_template = f"Relevant context: {results}\n\n The user's question: {question_def}"

    # print("++++++++++++++ with RAG")

    # input_ids_w_context = tokenizer(prompt_template, return_tensors="pt").to(device)
    # outputs = model.generate(**input_ids_w_context,
    #                         max_new_tokens=256)
    # print(tokenizer.decode(outputs[0]))

    # print("++++++++++++++ without RAG")

    # input_ids = tokenizer(f"The user's question: {question_def}", return_tensors="pt").to(device)
    # outputs = model.generate(**input_ids,
    #                         max_new_tokens=256)
    # print(tokenizer.decode(outputs[0]))

# 1) Build FAISS index as before:
def build_vectorstore(docs):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_documents(docs, embedding=embed)

# 2) Get top-M candidates with their similarity scores
def get_candidates(vectorstore, query, M=5):
    # returns list of (Document, sim_score)
    return vectorstore.similarity_search_with_relevance_scores(query, k=M)

# 3) Simulated buyer–value: v_i = sim_i (just take sim of each snippet)
def evaluate_performance_sim(candidates, k=1):
    # only top-k actually “used”
    # we’ll define V = sum of sims of the chosen k
    sims = [sim for (_, sim) in candidates]
    return sum(sims[:k])

# 4) Leave‑One‑Out payments
def loo_payments(query, vectorstore, docs, M=5, k=1):
    # 1) retrieve full candidate list
    full_cands = get_candidates(vectorstore, query, M=M)
    Vf = evaluate_performance_sim(full_cands, k)
    payments = {}
    # for each candidate j, remove it and recompute
    for j_idx, (doc_j, sim_j) in enumerate(full_cands):
        # build D_minus by dropping position j_idx
        minus = full_cands[:j_idx] + full_cands[j_idx+1:]
        Vminus = evaluate_performance_sim(minus, k)
        payments[doc_j.metadata.get("id", j_idx)] = Vf - Vminus
    return payments

# 5) Shapley payments (Monte‑Carlo)
def shapley_payments(query, vectorstore, docs, M=5, k=1, samples=200):
    full_cands = get_candidates(vectorstore, query, M=M)
    n = len(full_cands)
    shap = {doc.metadata.get("id", i): 0.0 for i, (doc, _) in enumerate(full_cands)}
    for _ in range(samples):
        perm = full_cands.copy()
        random.shuffle(perm)
        Vprev = 0.0
        prefix = []
        for (doc_j, sim_j) in perm:
            prefix_with = prefix + [(doc_j, sim_j)]
            Vwith = evaluate_performance_sim(prefix_with, k)
            marginal = Vwith - Vprev
            idx = doc_j.metadata.get("id")
            shap[idx] += marginal
            Vprev = Vwith
            prefix = prefix_with
    # average
    return {j: val/samples for j, val in shap.items()}

# 6) Demo on one query:
if __name__=="__main__":
    # load your QA “snippets” as docs with IDs and prices
    from datasets import load_dataset
    import pandas as pd

    df = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train").to_pandas()
    df["id"] = df.index
    docs = [Document(page_content=row["Answer"],
                     metadata={"id": row["id"], "price": random.random()*0.05})
            for _, row in df.head(500).iterrows()]

    # build index
    vs = build_vectorstore(docs)

    q = "What is a Sydenham chorea?"

    loo = loo_payments(q, vs, docs, M=5, k=1)
    shap = shapley_payments(q, vs, docs, M=5, k=1, samples=200)

    print("LOO payments:", loo)
    print("Shapley payments:", shap)
