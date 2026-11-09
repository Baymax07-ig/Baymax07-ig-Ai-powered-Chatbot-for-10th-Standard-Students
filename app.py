from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()


PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

embeddings = download_embeddings()


index_name="10th-chatbot"





docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)



retriever=docsearch.as_retriever(search_kwargs={"k": 3},search_type="similarity",)

chatModel= ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    GOOGLE_API_KEY= GOOGLE_API_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain=create_stuff_documents_chain(chatModel, prompt=prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)


@app.route("/")
def index():
    return render_template("chatbot.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg= request.form["msg"]
    input = msg
    print(input)
    response= rag_chain.invoke({"input":msg})
    print("response :",response["answer"])
    return str(response["answer"])
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)