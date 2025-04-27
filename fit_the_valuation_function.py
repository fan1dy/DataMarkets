import os
import sys
from dotenv import load_dotenv
from langchain.docstore.document import Document
from typing import List, Any
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
# from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
import argparse
from datasets import load_dataset
import random
import re    
import torch
from torch import cuda
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from together import Together
client = Together() # auth defaults to os.environ.get("TOGETHER_API_KEY")


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["TOGETHER_API_KEY"] = os.getenv('TOGETHER_API_KEY')




JUDGE_PROMPT = """
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
        return initial_docs[0], similarity_scores[0]

def rag_run(query: str, docs: List[Document]) -> None:
    # Load embedding model
    advanced_docs, score = custom_retriever._get_relevant_documents(query, num_docs=1)
    return advanced_docs, score

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--query", type=str, default='What are the impacts of climate change?', help="Query to ask")
    parser.add_argument("--retriever_type", type=str, default="reranker", choices=["reranker", "cross_encoder"],
                        help="Type of retriever to use")
    return parser.parse_args()

def extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    args = parse_args()
    data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
    data = data.to_pandas()
    data["id"]=data.index
    MAX_ROWS = len(data["id"])
    DOCUMENT="Answer"
    TOPIC="qtype"
    subset_data = data.head(MAX_ROWS)
    doc_lists = subset_data["Answer"].tolist()
    price_list = [random.uniform(0, 0.05) for _ in range(MAX_ROWS)]
    docs = [Document(page_content=doc_lists[i], metadata={"price": price_list[i]}) for i in range(len(doc_lists))]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    custom_retriever = CustomRetriever(vectorstore=vectorstore)

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # model_id = "Qwen/Qwen2.5-3B"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id,
    #                                             device_map=device,
    #                                             torch_dtype=torch.bfloat16)
    repo_id = "deepseek-ai/DeepSeek-R1"

    # llm_client = InferenceClient(
    #     model=repo_id,
    #     timeout=120,
    # )
    relevance_scores = []
    model_outputs = []
    # df = pd.read_csv('data/ground-truth-retrieval.csv')
    # for row in tqdm(df['question']):
    #     question_def = row
    #     results, score = rag_run(query=question_def, docs=docs)
    #     prompt_template = f"Relevant context: {results}\n\n The user's question: {question_def}"

    #     input_ids_w_context = tokenizer(prompt_template, return_tensors="pt").to(device)
    #     outputs = model.generate(**input_ids_w_context,
    #                             max_new_tokens=256)
    #     relevance_scores.append(score)
    #     output_text = tokenizer.decode(outputs[0])
    #     model_outputs.append(output_text)
    
    
    # df["model_output"] = model_outputs
    # df["relevance_score"] = relevance_scores
    # df.to_csv('data/query_w_llm_generated_ans.csv', index=False)
    df = pd.read_csv('data/query_w_llm_generated_ans.csv')
    judge_texts = []
    judge_scores = []
    for i in tqdm(range(len(df['question']))):
    #     judge_text = llm_client.text_generation(
    #     prompt=JUDGE_PROMPT.format(question=df["question"][i], answer=df["model_output"][i]),
    #     max_new_tokens=1000,
    # )
        response = client.chat.completions.create(
        model=repo_id ,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=df["question"][i], answer=df["model_output"][i])}]
        )
        judge_text = response.choices[0].message.content
        judge_texts.append(judge_text)
        judge_score = extract_judge_score(judge_text)
        judge_scores.append(judge_score)
    df['llm_judge_text']  = judge_texts
    df["llm_judge_score"] = judge_scores
    # df["llm_judge_score"] = (df["llm_judge_score"] / 10) + 1
    df.to_csv('data/query_w_llm_generated_ans_with_judge_scores.csv', index=False)

