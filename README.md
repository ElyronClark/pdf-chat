# PDF Chat

Upload any PDF and ask questions about it using AI. Built with LangChain, ChromaDB, and OpenAI.

## Features
- Upload any PDF via REST API
- Semantic search with ChromaDB vector store
- Answers grounded in document content only
- FastAPI backend with auto-generated docs

## Setup

1. Clone the repo
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file:
   - `OPENAI_API_KEY=your_openai_key_here`
6. Run: `uvicorn api:app --reload`
7. API docs: `http://127.0.0.1:8000/docs`

## Stack
- Python 3
- LangChain
- ChromaDB
- OpenAI Embeddings + GPT-4o-mini
- FastAPI
- Uvicorn