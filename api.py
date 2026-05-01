import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

vectorstore = None

class Question(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore
    
    # Save uploaded file
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Index the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    
    return {"message": f"Indexed {len(chunks)} chunks from {file.filename}"}

@app.post("/ask")
async def ask(body: Question):
    global vectorstore
    if not vectorstore:
        return {"error": "No PDF uploaded yet"}
    
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(body.question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(content=f"Answer the question based on this context:\n\n{context}"),
        HumanMessage(content=body.question)
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}

@app.get("/")
async def root():
    return {"message": "PDF Chat API is running"}