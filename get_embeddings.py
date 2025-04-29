import os
import sys
from dotenv import load_dotenv
from langchain.docstore.document import Document
from typing import List, Any

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
# from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
import argparse
from datasets import load_dataset
import random
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
random.seed(42)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Load embedding model
data_csv = pd.read_csv('data/medquad.csv')
docs = [
    Document(
    page_content=row["answer"],
    metadata={
        "id":         int(row["id"]),
        "price":      random.uniform(0, 0.1),
        "question":   row["question"],
        "answer":     row["answer"],
        "source":     row["source"],
        "focus_area": row["focus_area"]
    }
    )
    for _,row in data_csv.iterrows() if isinstance(row["answer"], str) 
    ]

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# Create FAISS vectorstore from documents
vectorstore = FAISS.from_documents(docs, embedding=embeddings)
vectorstore.save_local("faiss_index")