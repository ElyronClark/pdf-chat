import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

load_dotenv()

def load_and_index_pdf(pdf_path):
    # Step 1: Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    
    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Step 3: Embed and store in ChromaDB
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    print("Indexed in ChromaDB")
    
    return vectorstore

def ask_question(vectorstore, question):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=f"Answer the question based on this context:\n\n{context}"),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    pdf_path = "ugc_pdf_test.pdf"
    
    print("Loading and indexing PDF...")
    vectorstore = load_and_index_pdf(pdf_path)
    
    print("\nAsk questions about the PDF (type 'quit' to exit)\n")
    while True:
        question = input("You: ")
        if question.lower() == 'quit':
            break
        answer = ask_question(vectorstore, question)
        print(f"Answer: {answer}\n")