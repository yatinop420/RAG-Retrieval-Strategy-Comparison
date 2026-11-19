# RAG Retrieval Strategy Comparison System

A Retrieval-Augmented Generation (RAG) application that compares five different retrieval strategies and generates context-aware answers using a Large Language Model.

## Overview

This project explores how different retrieval techniques perform when searching information from a Deep Learning PDF document.

Users can ask questions and compare the retrieved results and generated answers across five retrieval strategies through an interactive Streamlit interface.

##  Tech Stack

- **Python** – Core programming language
- **LangChain** – RAG pipeline and retrieval components
- **ChromaDB** – Vector database
- **Sentence Transformers** – Text embeddings
- **Groq API** – LLM-powered answer generation
- **Streamlit** – Interactive user interface
- **BM25** – Keyword-based retrieval

## 🔍 Retrieval Strategies

1. Similarity Search
2. Maximum Marginal Relevance (MMR)
3. Multi-Query Retrieval
4. BM25 Retrieval
5. Hybrid Search (Dense + Sparse Retrieval)

## ✨ Features

- PDF document ingestion and text chunking
- Semantic search using vector embeddings
- Multiple retrieval strategies
- AI-generated answers using retrieved context
- Quick Answer mode
- Compare All mode to compare retrieval strategies
- Retrieval and generation time display
- Retrieved source documents

## 📁 Project Structure

```text
├── app.py
├── config.py
├── ingestion.py
├── retrievers.py
├── generator.py
├── requirements.txt
├── documents/
│   └── deeplearning.pdf
└── .gitignore
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG-Retrieval-Strategy-Comparison
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Run document ingestion

```bash
python ingestion.py
```

### 6. Launch the application

```bash
streamlit run app.py
```

## 🎯 Project Objective

The objective is to understand and compare different retrieval approaches in RAG systems and observe how retrieval quality affects the final LLM-generated response.

---

**Built with Python, LangChain, ChromaDB and Groq.**