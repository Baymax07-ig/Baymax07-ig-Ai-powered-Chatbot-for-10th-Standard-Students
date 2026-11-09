import os
import time
from dotenv import load_dotenv
from langchain_openai import embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, filter_to_reduce_size, text_splitter, download_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
assert PINECONE_API_KEY, "PINECONE_API_KEY not found in .env"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"]= PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY




# 1. Load, filter, split
extracted_data = load_pdf_file("data/")
filtered_data = filter_to_reduce_size(extracted_data)
text_chunks = text_splitter(filtered_data)


embeddings = download_embeddings()
Pinecone_api_key= PINECONE_API_KEY
pc = Pinecone(api_key=Pinecone_api_key)



index_name = "10th-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
        
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)
print("Done. Chunks stored in Pinecone.")