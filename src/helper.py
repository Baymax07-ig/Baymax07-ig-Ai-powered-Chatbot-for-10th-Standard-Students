from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


#Extract data from the pdf file

def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents



def filter_to_reduce_size(documents: List[Document]) -> List[Document]:
    minimal_documents : List[Document] = []
    for doc in documents:
        src=doc.metadata.get("source")
        minimal_documents.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_documents


def text_splitter(minimal_documents):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    text_chunks=text_splitter.split_documents(minimal_documents)
    return text_chunks


def download_embeddings():
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
