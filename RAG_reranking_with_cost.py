# combined_rag_payments.py
import os
import sys
import random
import argparse
from dotenv import load_dotenv
from typing import Dict
import pandas as pd
import openai

from typing import List, Any
from pydantic import BaseModel, Field
from datasets import load_dataset

from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util

with open(".env", "w") as f:
  f.write("API Key")
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")

def rerank_documents(docs: List[Document], scores: List[float], top_n: int = 1) -> List[Document]:
    """Utility = sim - price reranking; return top_n Documents."""
    utilities = [scores[i] - docs[i].metadata["price"] for i in range(len(docs))]
    ranked = sorted(zip(docs, utilities), key=lambda x: x[1], reverse=True)
    return [doc for doc,_ in ranked[:top_n]]

class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, num_docs: int = 1) -> List[Document]:
        initial = self.vectorstore.similarity_search_with_relevance_scores(query, k=10)
        docs, sims = zip(*initial)
        print("Raw sims:", sims)
        return rerank_documents(list(docs), list(sims), top_n=num_docs)

def rag_run(query: str, docs: List[Document], top_n: int = 1) -> List[Document]:

    INDEX_DIR  = "faiss_index"
    embed = OpenAIEmbeddings(model="text-embedding-ada-002")

    if os.path.isdir(INDEX_DIR):
        # Load the existing FAISS index
        vectorstore = FAISS.load_local(INDEX_DIR, embed, allow_dangerous_deserialization=True)
        print(f"Loaded FAISS index from {INDEX_DIR}")
    else:
        # Build & save a new index
        vectorstore = FAISS.from_documents(docs, embedding=embed)
        vectorstore.save_local(INDEX_DIR)
        print(f"Saved FAISS index to {INDEX_DIR}")
    retriever  = CustomRetriever(vectorstore=vectorstore)
    return retriever._get_relevant_documents(query, num_docs=top_n)

def build_vectorstore(docs: List[Document]) -> FAISS:
    embed = OpenAIEmbeddings(model="text-embedding-ada-002")
    if os.path.isdir("my_index"):
        vs = FAISS.load_local("my_index", embed, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(docs, embedding=embed)
        vs.save_local("my_index")
    return vs

def get_candidates(vectorstore: FAISS, query: str, M: int = 5):

    return vectorstore.similarity_search_with_relevance_scores(query, k=M)

def evaluate_performance_sim(cands: List[tuple], k: int = 1):
    sims = [sim for (_, sim) in cands]
    return sum(sims[:k])

def evaluate_performance_real(query: str, true_answer: str, vs: FAISS, k: int = 1, model: str = "gpt-3.5-turbo") -> float:
    cands = vs.similarity_search(query, k=k)
    ctx = "\n\n".join(d.page_content for d in cands)

    prompt = (
        f"Use the following context to construct your answer:\n\n"
        f"{ctx}\n\n"
        f"Question: {query}\n"
        f"Answer (brief):"
    )

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    llm_ans = response.choices[0].message["content"].strip().split("\n")[0]
    print(llm_ans)

    shorbet = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = shorbet.encode([llm_ans, true_answer], convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    return cosine_score

def rag_yesno(query, vs, k=1, model="gpt-3.5-turbo"):
    cands = vs.similarity_search(query, k=k)
    ctx = "\n\n".join(d.page_content for d in cands)
    prompt = (
      f"Context:\n{ctx}\n\n"
      f"Question: {query}\n"
      f"Answer (yes or no):"
    )
    resp = openai.ChatCompletion.create(
      model=model,
      messages=[{"role":"user","content":prompt}],
      temperature=0
    )
    #out = resp.choices[0].message.content.lower()
    out = "yes"
    return "yes" if "yes" in out else "no"

def evaluate_performance_mcq(q, true_ans, vs, k=1):
    pred = rag_yesno(q, vs, k)
    return 1.0 if pred == true_ans else 0.0



def loo_payments_mcq(query: str, true_answer: str, vs: FAISS, M: int = 5, k: int = 1) -> Dict[int, float]:

    # 1) Retrieve M candidates
    full = get_candidates(vs, query, M)
    docs_M = [doc for doc, _ in full]
    # 2) Full utility
    V_full = evaluate_performance_mcq(query, true_answer, vs, k=k)
    payments = {}
    # 3) For each candidate j, drop it and recompute
    for idx, (doc_j, _) in enumerate(full):
        # Build a temporary FAISS on the remaining docs
        minus_docs = docs_M[:idx] + docs_M[idx+1:]
        temp_vs = FAISS.from_documents(minus_docs, embedding=vs.embedding)
        V_minus = evaluate_performance_mcq(query, true_answer, temp_vs, k=k)
        payments[doc_j.metadata["id"]] = V_full - V_minus
    return payments


def shapley_payments_mcq(query: str, true_answer: str, vs: FAISS, M: int = 5, k: int = 1, samples: int = 200) -> Dict[int, float]:

    full = get_candidates(vs, query, M)
    # initialize accumulators
    shap = {doc.metadata["id"]: 0.0 for doc, _ in full}
    for _ in range(samples):
        perm = full.copy()
        random.shuffle(perm)
        V_prev = 0.0
        prefix = []
        for (doc_j, _) in perm:
            prefix = prefix + [(doc_j, None)]
            # build temp_vs from prefix docs
            temp_vs = FAISS.from_documents([d for d,_ in prefix], embedding=vs.embedding)
            V_with = evaluate_performance_mcq(query, true_answer, temp_vs, k=k)
            marginal = V_with - V_prev
            shap[doc_j.metadata["id"]] += marginal
            V_prev = V_with
    # average
    return {j: shap[j] / samples for j in shap}


def myerson_payments_mcq(query: str, true_answer: str, vs: FAISS, M: int = 5, k: int = 1
) -> Dict[int, float]:

    full = get_candidates(vs, query, M)

    utils = {doc.metadata["id"]: (evaluate_performance_mcq(query, true_answer, vs, k=k) - doc.metadata["price"])
             for doc, _ in full}
    # sort descending
    sorted_ids = sorted(utils, key=utils.get, reverse=True)
    winners = sorted_ids[:k]
    # the “critical threshold” is the (k+1)-th utility
    threshold = utils[sorted_ids[k]] if len(sorted_ids) > k else 0.0

    payments = {}
    for j, u in utils.items():
        payments[j] = max(0.0, u - threshold) if j in winners else 0.0
    return payments


def parse_args():
    parser = argparse.ArgumentParser(description="RAG + Payment Simulation")
    parser.add_argument("--query", type=str, default="What is a Sydenham chorea?")
    parser.add_argument("--num_snippets", type=int, default=5)
    parser.add_argument("--loo_k", type=int, default=1)
    parser.add_argument("--shap_samples", type=int, default=200)
    # use parse_known_args to swallow Jupyter’s -f/--profile flags
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    #df = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train").to_pandas()
    #df["id"] = df.index
    #docs = [
    #  Document(page_content=row["Answer"],
    #           metadata={"id": int(row["id"]), "price": random.random() * 0.05})
    #  for _, row in df.head(500).iterrows()
    #]
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="test")
    docs = [
        Document(
            page_content=ds[i]["context"],
            metadata={
                "id": i,
                "price": random.random() * 0.05,
                "true_answer": "yes" if ds[i]["label"] == 1 else "no"
            }
        )
        for i in range(len(ds))
    ]

    print("\nRAG Retrieval")
    retrieved = rag_run(args.query, docs, top_n=1)

    for doc in retrieved:
        print("Retrieved snippet:", doc.page_content[:200], "...")

    vs = build_vectorstore(docs)

    questionID = 123
    true_ans = next(doc.metadata["true_answer"]
                    for doc in docs if doc.metadata["id"] == questionID)

    loo  = loo_payments_mcq(args.query, true_ans, vs, M=args.num_snippets, k=args.loo_k)
    shap = shapley_payments_mcq(args.query, true_ans, vs,
                                M=args.num_snippets, k=args.loo_k,
                                samples=args.shap_samples)
    myer = myerson_payments_mcq(args.query, true_ans, vs,
                                M=args.num_snippets, k=args.loo_k)
    #VCG payment

    print("\n Simulated Payments")
    print("LOO payments:", loo)
    print("Shapley payments:", shap)
    print("Myer payments:", myer)

    #insert plot, different x values, which is reported price that maximizes sellers utility. follow from the Figure 1

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
    # print(tokenizer.decode(outputs[0])) #answer from Qwen

    # judge this answer from high end LLM


    # print("++++++++++++++ without RAG")

    # input_ids = tokenizer(f"The user's question: {question_def}", return_tensors="pt").to(device)
    # outputs = model.generate(**input_ids,
    #                         max_new_tokens=256)
    # print(tokenizer.decode(outputs[0]))
