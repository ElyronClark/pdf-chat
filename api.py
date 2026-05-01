import os
import shutil
import uuid
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

vectorstores = {}  # store multiple PDFs

class Question(BaseModel):
    question: str
    session_id: str  # which PDF to query

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    
    file_path = f"uploads/{session_id}_{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstores[session_id] = Chroma.from_documents(chunks, embeddings)
    
    return {"message": f"Indexed {len(chunks)} chunks from {file.filename}", "session_id": session_id}

@app.post("/ask")
async def ask(body: Question):
    if body.session_id not in vectorstores:
        return {"error": "PDF not found. Please upload again."}
    
    retriever = vectorstores[body.session_id].as_retriever()
    docs = retriever.invoke(body.question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(content=f"Answer the question based only on this context. If the answer isn't in the context, say so.\n\n{context}"),
        HumanMessage(content=body.question)
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}