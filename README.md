# PDF Chat

Upload any PDF and ask questions about it. Built with LangChain, ChromaDB, and OpenAI.

## Features
- Load and index any PDF
- Semantic search with ChromaDB vector store
- Answers grounded in document content
- FastAPI backend (coming soon)
- React UI (coming soon)

## Setup

1. Clone the repo
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file:
   - `OPENAI_API_KEY=your_openai_key_here`
6. Run: `python3 main.py`

## Stack
- Python 3
- LangChain
- ChromaDB
- OpenAI Embeddings + GPT-4o-mini
- FastAPI (coming soon)