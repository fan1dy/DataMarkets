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
