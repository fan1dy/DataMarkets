# combined_rag_payments.py
import os
import sys
import random
import argparse
import json
import re
from dotenv import load_dotenv
from typing import Dict
from tqdm import tqdm
from together import Together
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

#with open(".env", "w") as f:
#  f.write("API Key")
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] ="insert key"
os.environ["TOGETHER_API_KEY"] = "insert key"

openai.api_key = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=API_KEY, timeout=10)


prompt_template = """
You emulate a user of our medical QA system.
Formulate 4 questions this user might ask based on a provided disease record.
Make them specific to this disease. Use as few words as possible from the record.

The record:
question: {question}
answer: {answer}
source: {source}
focus_area: {focus_area}

Return JSON:
{{"questions": ["q1","q2","q3","q4"]}}
""".strip()

def generate_questions(record: Dict[str,Any]) -> List[str]:
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": prompt_template.format(**record)}],
        temperature=0
    )
    return json.loads(resp.choices[0].message.content)["questions"]


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

def rag_run(query: str, docs: List[Document], top_n: int = 1):
    INDEX_DIR  = "faiss_index"
    embed = OpenAIEmbeddings(model="text-embedding-3-large")

    if os.path.isdir(INDEX_DIR):
        # Load the existing FAISS index
        vectorstore = FAISS.load_local(INDEX_DIR, embed, allow_dangerous_deserialization=True)
        print(f"Loaded FAISS index from {INDEX_DIR}")
    else:

        vectorstore = FAISS.from_documents(docs, embedding=embed)
        vectorstore.save_local(INDEX_DIR)
        print(f"Saved FAISS index to {INDEX_DIR}")

    vectorstore.embedding = embed

    retriever  = CustomRetriever(vectorstore=vectorstore)
    return retriever._get_relevant_documents(query, num_docs=top_n)

def build_vectorstore(docs: List[Document]) -> FAISS:
    embed = OpenAIEmbeddings(model="text-embedding-3-large")
    if os.path.isdir("my_index"):
        vs = FAISS.load_local("my_index", embed, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(docs, embedding=embed)
        vs.save_local("my_index")
    vs.embedding = embed
    return vs

def get_candidates(vectorstore: FAISS, query: str, M: int = 5):

    return vectorstore.similarity_search_with_relevance_scores(query, k=M)


def extract_judge_score(answer: str, split_str: str = "Total rating:") -> float:
    m = re.search(rf"{split_str}\s*([0-9]+(?:\.[0-9]+)?)", answer)
    if not m:
        return 0.0
    return float(m.group(1))

def evaluate_performance_judge(question: str, vs: FAISS, k: int = 1, model: str = "deepseek-ai/DeepSeek-R1"):
    JdgPrompt="""
    You will be given a user_question and system_answer couple.
    Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
    Give your answer as a float on a scale of 0 to 10, where 0 means that the system_answer is not helpful at all, and 10 means that the answer completely and helpfully addresses the question.

    Provide your feedback as follows:

    Feedback:::
    Total rating: (your rating, as a float between 0 and 10)

    Now here are the question and answer.

    Question: {question}
    Answer: {answer}

    Feedback:::
    Total rating: """

    docs_and_sims = vs.similarity_search_with_relevance_scores(question, k=k)
    snippet, temp = docs_and_sims[0]

    prompt = JdgPrompt.format(question=question, answer=snippet.page_content)

    '''resp = together.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        timeout=20
    )'''

    models = client.models.list() 

    resp = client.chat.completions.create(
        model=models[0].id,
        messages=[{"role": "user", "content": prompt}],
        timeout = 30,
        max_retries = 2
    )

    judge_text = resp.choices[0].message.content

    raw_score = extract_judge_score(judge_text)   # e.g. “7.5”
    if raw_score is None:
        return 0.0
    return float(raw_score) / 10.0  # now in [0,1]


def loo_payments(query: str, true_answer: str, vs: FAISS, docs: List[Document], M: int = 5, k: int = 1):
    V_full = evaluate_performance_judge(query, vs, k=k)

    full = get_candidates(vs, query, M)
    docs_M = [doc for doc,_ in full]

    payments: Dict[int, float] = {}
    for idx, (doc_j, _) in enumerate(full):
        # rebuild tiny index w/o doc_j
        minus_docs = [d for d in docs_M if d.metadata["id"] != doc_j.metadata["id"]]
        temp_vs = FAISS.from_documents(minus_docs, embedding=vs.embedding)
        V_minus = evaluate_performance_judge(query, temp_vs, k=k)
        payments[doc_j.metadata["id"]] = V_full - V_minus

    return payments


def shapley_payments(query: str, true_answer: str, vs: FAISS, docs: List[Document], M: int = 5, k: int = 1, samples: int = 200):
    full = get_candidates(vs, query, M)
    shap = {doc.metadata["id"]: 0.0 for doc,_ in full}

    for tmp in range(samples):
        perm = full.copy()
        random.shuffle(perm)
        V_prev = 0.0
        prefix_docs: List[Document] = []

        for doc_j, temp in perm:
            prefix_docs.append(doc_j)
            temp_vs = FAISS.from_documents(prefix_docs, embedding=vs.embedding)
            V_with = evaluate_performance_judge(query, temp_vs, k=k)
            shap[doc_j.metadata["id"]] += (V_with - V_prev)
            V_prev = V_with

    return {j: shap[j] / samples for j in shap} #Dict[int, float]


def myerson_payments(query: str, true_answer: str, vs: FAISS, M: int = 5, k: int = 1):

    full = get_candidates(vs, query, M)
    v_single = {}
    for (doc_j, _) in full:
        tmp_vs = FAISS.from_documents([doc_j], embedding=vs.embedding)
        v_single[doc_j.metadata["id"]] = evaluate_performance_judge(query, tmp_vs, k=1)

    utils = {j: v_single[j] - doc.metadata["price"]
             for doc,j in zip([d for d,tmp in full], v_single.keys())}
    sorted_ids = sorted(utils, key=utils.get, reverse=True)
    winners   = sorted_ids[:k]
    threshold = utils[sorted_ids[k]] if len(sorted_ids)>k else 0.0

    return {j: max(0.0, utils[j]-threshold) if j in winners else 0.0 for j in utils}

def vcg_payments(query: str, vs: FAISS, docs: List[Document], M: int = 5,k: int = 1):
    # 1) top-M by similarity
    cands = get_candidates(vs, query, M) 
    sims   = {doc.metadata["id"]: sim  for doc, sim  in cands}
    costs  = {doc.metadata["id"]: doc.metadata["price"] for doc,_ in cands}
    utils  = {j: sims[j] - costs[j]       for j in sims}

    # 2) rank by utility and pick winners S
    ranked      = sorted(utils.items(), key=lambda x: x[1], reverse=True)
    winners_ids = [j for j, _ in ranked[:k]]

    # 3) precompute sum of "others in S" for each winner
    sum_others = {
      j: sum(utils[i] for i in winners_ids if i != j)
      for j in winners_ids
    }

    payments: Dict[int, float] = {}
    for j in utils:
        if j not in winners_ids:
            payments[j] = 0.0
        else:
            # build the M−1 candidate list without j
            others = [(i, utils[i]) for i in utils if i != j]
            # re-rank them and take top-k
            best_k = sorted(others, key=lambda x: x[1], reverse=True)[:k]
            h_j    = sum(u for _, u in best_k)
            payments[j] = max(0.0, h_j - sum_others[j])
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

    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset")
    df = ds["train"].to_pandas().reset_index()
    df.rename(columns={"index":"id"},inplace=True)

    docs = [
      Document(
        page_content=row["Answer"],
        metadata={
          "id":         int(row["id"]),
          "price":      random.random()*0.05,
          "question":   row["Question"],
          "answer":     row["Answer"],
          "source":     row.get("Source",""),
          "focus_area": row.get("Topic","")
        }
      )
      for _,row in df.iterrows()
    ]
    #ds = load_dataset("pubmed_qa", "pqa_labeled", split="test")
    #docs = [
    #    Document(
    #        page_content=ds[i]["context"],
    #        metadata={
    #            "id": i,
    #            "price": random.random() * 0.05,
    #            "true_answer": "yes" if ds[i]["label"] == 1 else "no"
    #        }
    #    )
    #    for i in range(len(ds))
    #]

    questionID = 7368
    record = docs[questionID].metadata
    questions   = generate_questions(record)
    query       = questions[0]
    true_answer = record["answer"]

    print(f"question: {query!r}")
    print(f"Ground-truth answer: {true_answer!r}")

    print("\nRAG Retrieval")
    retrieved = rag_run(query, docs, top_n=1)
    for doc in retrieved:
        print("Retrieved snippet:", doc.page_content[:200], "…")

    '''true_ans = next(doc.metadata["true_answer"]
                    for doc in docs if doc.metadata["id"] == questionID)

    loo  = loo_payments_mcq(args.query, true_ans, vs, M=args.num_snippets, k=args.loo_k)
    shap = shapley_payments_mcq(args.query, true_ans, vs, M=args.num_snippets, k=args.loo_k, samples=args.shap_samples)
    myer = myerson_payments_mcq(args.query, true_ans, vs, M=args.num_snippets, k=args.loo_k)'''

    vs = build_vectorstore(docs)

    myers = myerson_payments(query, true_answer, vs, M=args.num_snippets, k=args.loo_k)
    vcg = vcg_payments(query, vs, docs, M=args.num_snippets, k=args.loo_k)
    loo = loo_payments(query, true_answer, vs, docs, M=args.num_snippets, k=args.loo_k)
    shap= shapley_payments(query, true_answer,vs, docs, M=args.num_snippets, k=args.loo_k, samples=args.shap_samples)
    

    #real_v = evaluate_performance_judge(query, true_answer, vs, k=args.loo_k)
    #VCG payment

    print("\n Simulated Payments")
    print("VCG payments:", vcg)
    print("LOO payments:", loo)
    print("Shapley payments:", shap)
    print("Myer payments:", myers)

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
