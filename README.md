# Conversational RAG System

A modular, production-grade Conversational Retrieval-Augmented Generation (RAG) pipeline built using Python, LangChain, ChromaDB, and Google Gemini.

## Features

- **Deterministic Ingestion:** Recursively chunks documents and generates unique IDs to prevent duplicate embeddings in ChromaDB.
- **Vector Retrieval:** Semantic search using HuggingFace (`all-MiniLM-L6-v2`) embeddings with similarity thresholding.
- **Dynamic Query Reformulation:** Rewrites ambiguous follow-up questions using recent chat context before executing vector searches.
- **Sliding-Window Memory:** Retains the last 6 conversation turns to preserve context while optimizing token costs.
- **Groq Integration:** Uses `openai/gpt-oss-120b` via `langchain-groq` for low-latency generation.

---

## Project Structure

```text
├── docs/                     # Sample text knowledge base files
├── Ingestion_Pipeline.py     # Document loading, chunking, and ChromaDB upsert logic
├── Retrieval_Pipeline.py     # Vector retrieval and similarity search
├── answer.py                 # Query reformulation, sliding memory, and generation loop
├── app.py                    # Streamlit app
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusions
├── requirements.txt          # Project Dependencies
└── README.md
