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
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Load embedding model
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
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# Create FAISS vectorstore from documents
vectorstore = FAISS.from_documents(docs, embedding=embeddings)
vectorstore.save_local("faiss_index")
