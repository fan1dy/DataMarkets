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
from langchain_community.vectorstores import FAISS
from typing import List, Any
from pydantic import BaseModel, Field
from datasets import load_dataset

from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings

from langchain_core.retrievers import BaseRetriever

#with open(".env", "w") as f:
#  f.write("API Key")
from dotenv import load_dotenv
import os
from together import Together
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import cuda

import random
import numpy as np

# fix random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)          # For single-GPU
torch.cuda.manual_seed_all(42)  

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print("use device:", device)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["TOGETHER_API_KEY"] = os.getenv('TOGETHER_API_KEY')

client = Together() 
model_id = "Qwen/Qwen2.5-3B"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                            device_map=device,
                                            torch_dtype=torch.bfloat16)

INDEX_DIR  = "/iopsstor/scratch/cscs/dfan/RAG/DataMarkets-proj/faiss_index" # change it to your local faiss path

question_prompt_template =  """
You emulate a user of our medical question answering application.
Formulate 4 questions this user might ask based on a provided disease.
Make the questions specific to this disease.
The record should contain the answer to the questions, and the questions should
be complete and not too short. Use as few words as possible from the record. 

The record:

question: {question}
answer: {answer}
source: {source}
focus_area: {focus_area}

Provide the output in parsable JSON without using code blocks:

{{"questions": ["question1", "question2", ..., "question4"]}}
""".strip()

def generate_questions(record: Dict[str,Any]) -> List[str]:

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": question_prompt_template.format(**record)}],
        temperature=0
    )
    questions = json.loads(resp.choices[0].message.content)["questions"]
    return questions 

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")

def rerank_documents(docs: List[Document], scores: List[float], top_n: int = 1) -> List[Document]:
    """Utility = sim - price reranking; return top_n Documents."""
    utilities = [scores[i] - docs[i].metadata["price"] for i in range(len(docs))]
    ranked = sorted(zip(docs, utilities), key=lambda x: x[1], reverse=True)
    return [doc for doc,_ in ranked[:top_n]]

def rerank_documents_with_LLM(docs: List[Document], scores: List[float], query: str, top_n: int = 1) -> List[Document]:
    """Utility = LLM score - price reranking; return top_n Documents."""
    scores = [evaluate_performance_judge(query, docs[i].page_content, k=1) for i in range(len(docs))]
    prices = [docs[i].metadata["price"] for i in range(len(docs))]
    utilities = [scores[i] - docs[i].metadata["price"] for i in range(len(docs))]
    ranked = sorted(zip(docs, utilities), key=lambda x: x[1], reverse=True)
    doc_ids = [doc.metadata["id"] for doc in docs]
    print("prices:", [docs[i].metadata["price"] for i in range(len(docs))])
    print("scores:",scores)
    return prices, scores, doc_ids

class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, num_docs: int = 1, initial_docs: int=10) -> List[Document]:
        initial = self.vectorstore.similarity_search_with_relevance_scores(query, k=initial_docs)
        docs, sims = zip(*initial)
        # return rerank_documents(docs,sims, top_n=num_docs)
        return rerank_documents_with_LLM(docs, sims, query, top_n=num_docs)

def rag_run(query: str, vectorstore: FAISS, top_n: int = 1, initial_docs: int=10):

    retriever  = CustomRetriever(vectorstore=vectorstore)
    return retriever._get_relevant_documents(query, num_docs=top_n, initial_docs=initial_docs)

def build_vectorstore(docs: List[Document]) -> FAISS:
    embed = OpenAIEmbeddings(model="text-embedding-3-large")
    if os.path.isdir(INDEX_DIR):
        vs = FAISS.load_local(INDEX_DIR, embed, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(docs, embedding=embed)
        vs.save_local(INDEX_DIR)
    vs.embedding = embed
    return vs

def get_candidates(vectorstore: FAISS, query: str, M: int = 5):

    return vectorstore.similarity_search_with_relevance_scores(query, k=M)


def extract_judge_score(answer: str, split_str: str = "Total rating:") -> float:
    m = re.search(rf"{split_str}\s*([0-9]+(?:\.[0-9]+)?)", answer)
    if not m:
        return 0.0
    return float(m.group(1))

def evaluate_performance_judge(question: str, retrived: str, k: int = 1, model: str = "deepseek-ai/DeepSeek-R1"):
    # generate answer based on the retrieced doc first
    answer = get_llm_answer_from_retrieved_doc(retrived, question)
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

    prompt = JdgPrompt.format(question=question, answer=answer)

    '''resp = together.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        timeout=20
    )'''

    # models = client.models.list() 

    resp = client.chat.completions.create(
        # model=models[0].id,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout = 30,
        max_retries = 2
    )

    judge_text = resp.choices[0].message.content

    raw_score = extract_judge_score(judge_text)   # e.g. “7.5”
    if raw_score is None:
        return 0.0
    print("Judge response:", raw_score)
    return float(raw_score) / 10.0  # now in [0,1]

def get_llm_answer_from_retrieved_doc(retrieved: str, query: str):
    prompt_template = f"Relevant context: {retrieved}\n\n The user's question: {query}"
    input_ids_w_context = tokenizer(prompt_template, return_tensors="pt").to(device)
    outputs = model.generate(**input_ids_w_context,
                            max_new_tokens=256)
    llm_answer = tokenizer.decode(outputs[0])
    return llm_answer

def loo_payments(prices, scores, doc_ids):
    """
    LOO payment is the difference in valuations when seller j is removed from the market.
    """
    utility =  [scores[i] - prices[i] for i in range(len(prices))]
    idx = utility.index(max(utility))
    v_full = scores[idx]
    payments: Dict[int, float] = {}
    for id in range(len(doc_ids)):
        prices_tmp = prices.copy()
        scores_tmp = scores.copy()
        prices_tmp.pop(id)
        scores_tmp.pop(id)
        # rebuild tiny index w/o doc_j
        utility_minus_j = [-prices_tmp[i] + scores_tmp[i] for i in range(len(prices_tmp))]
        id_ = utility_minus_j.index(max(utility_minus_j))
        payments[int(doc_ids[id])] = v_full - scores_tmp[id_]
    return payments


def shapley_payments(prices, scores, doc_ids):
    """
    LOO payment is the average difference in valuations when seller j is removed from any subset the market.
    """
    from math import comb
    from itertools import chain, combinations
    utility = [scores[i] - prices[i] for i in range(len(prices))]
    payments: Dict[int, float] = {}
    elements = list(range(len(doc_ids)))

    def all_subsets(iterable):
        return chain.from_iterable(combinations(iterable, r) for r in range(len(iterable) + 1))
    
    for id in range(len(doc_ids)):
        elements_tmp = elements.copy()
        elements_tmp.pop(id)
        subsets = list(map(list, all_subsets(elements_tmp)))
        Shapley_value = 0
        for subset in subsets:
            if len(subset) >0:
                subset_plus_j = subset + [id]
                utility_subset = [utility[i] for i in subset]
                utility_subset_plus_j = [utility[i] for i in subset_plus_j]
                idx1 = subset_plus_j[utility_subset.index(max(utility_subset))]
                idx2 = subset_plus_j[utility_subset_plus_j.index(max(utility_subset_plus_j))]
                Shapley_value += (1/comb(len(doc_ids)-1,len(subset))) * (scores[idx2]-scores[idx1])
        payments[int(doc_ids[id])] = Shapley_value/len(doc_ids)

    return payments


def myerson_payments(prices, scores, doc_ids):
    payments: Dict[int, float] = {}
    for id in doc_ids:
        payments[int(id)] = 0.0
    utility =  [scores[i] - prices[i] for i in range(len(prices))]
    idx = utility.index(max(utility))
    cj = prices[idx]
    # now calculate the residual part
    sorted_utility = sorted(utility, reverse=True)
    res = sorted_utility[0] -sorted_utility[1] - cj
    payments[int(doc_ids[idx])] =  res + cj
    return payments

def vcg_payments(prices, scores, doc_ids):
    """
    VCG payment is the average difference in utlities when seller j is removed from any subset the market.
    """
    payments: Dict[int, float] = {}
    utility =  [scores[i] - prices[i] for i in range(len(prices))]
    idx = utility.index(max(utility))
    max_u = utility[idx]
    for id in range(len(doc_ids)):
        utility_tmp = utility.copy()
        utility_tmp.pop(id)
        sorted_utility_minus_j = sorted(utility_tmp, reverse=True)
        payments[int(doc_ids[id])] = max_u - sorted_utility_minus_j[0] 
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
    df = pd.read_csv('data/medquad.csv')
    # df = pd.read_csv('../DataMarkets-proj/data/medquad.csv')
    df = df.sample(n=50)
    docs = [
      Document(
        page_content=row["answer"],
        metadata={
          "id":         int(row["id"]),
          "price":      random.random()*0.05,
          "question":   row["question"],
          "answer":     row["answer"],
          "source":     row.get("source",""),
          "focus_area": row.get("focus_area","")
        }
      )
      for _,row in df.iterrows()
    ]

    questionID = 7#1
    record = docs[questionID].metadata
    questions   = generate_questions(record)
    query       = questions[0]
    true_answer = record["answer"]

    print(f"question: {query!r}")
    print(f"Ground-truth doc: {true_answer!r}")


    vs = build_vectorstore(docs)
    print("\nRAG Retrieval")
    prices, scores, doc_ids = rag_run(query, vs, top_n=1,initial_docs=10)

    results = {}
    results["prices"] = [float(ele) for ele in prices]
    results["scores"] = [float(ele) for ele in scores]
    results["doc_ids"] = [int(ele) for ele in doc_ids]

    results["query"] = query
    ret = vs.similarity_search_with_relevance_scores(query, k=1)
    docs, sims = zip(*ret)
    results["true_doc_id"] = int(docs[0].metadata["id"])

    myers = myerson_payments(prices, scores, doc_ids)
    print("Myer payments:", myers)
    vcg = vcg_payments(prices, scores, doc_ids)
    print("VCG payments:", vcg)
    loo = loo_payments(prices, scores, doc_ids)
    print("LOO payments:", loo)
    shap= shapley_payments(prices, scores, doc_ids)
    print("Shapley payments:", shap)

    results["Myerson"] = myers
    results["VCG"] = vcg
    results["LOO"] = loo
    results["Shapley"] = shap
    
    with open("res.json", "w") as f:
        json.dump(results, f, indent=4)  # `indent=4` makes it nicely formatted