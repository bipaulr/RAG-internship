# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot developed as part of a Product Engineering Internship. The project demonstrates how Large Language Models can answer questions using external documents instead of relying only on pretrained knowledge.

The chatbot retrieves relevant information from a custom document collection using semantic search and generates grounded responses using Groq's Llama 3.3 70B model.

---

## Project Overview

This project follows the RAG pipeline introduced during the internship:

1. Load documents
2. Split documents into smaller chunks
3. Generate embeddings for every chunk
4. Store embeddings in ChromaDB
5. Embed the user's question
6. Retrieve the most relevant chunks
7. Generate an answer using the retrieved context
8. Display the answer and retrieved sources through a Streamlit interface

---

## Features

- Document ingestion pipeline
- Automatic text chunking
- Local embedding generation using Sentence Transformers
- ChromaDB vector database
- Semantic similarity search
- Question answering using Groq Llama 3.3 70B
- Streamlit-based chat interface
- Displays retrieved document chunks and source files

---

## Architecture

```
                    Documents
                        │
                        ▼
              Text Document Loader
                        │
                        ▼
          Recursive Character Splitter
                        │
                        ▼
        Sentence Transformer Embeddings
                        │
                        ▼
                ChromaDB Vector Store

────────────────────────────────────────────

                 User Question
                        │
                        ▼
            Sentence Transformer
                        │
                        ▼
             Similarity Search (Top K)
                        │
                        ▼
             Retrieved Document Chunks
                        │
                        ▼
                Groq Llama 3.3 70B
                        │
                        ▼
                  Generated Answer
                        │
                        ▼
                  Streamlit Interface
```

---

## Technologies Used

### Programming Language

- Python

### Frameworks & Libraries

- Streamlit
- LangChain
- LangChain Chroma
- LangChain HuggingFace
- Sentence Transformers
- python-dotenv

### Vector Database

- ChromaDB

### Embedding Model

- sentence-transformers/all-MiniLM-L6-v2

### Large Language Model

- Groq
- Llama 3.3 70B Versatile

---

## Project Structure

```
RAG-internship
│
├── app.py
├── ingest.py
├── query.py
├── rag_pipeline.py
├── documents/
├── chroma_db/
├── requirements.txt
└── README.md
```

---

## Workflow

### Document Ingestion

The ingestion pipeline performs the following steps:

- Reads all text documents from the `documents` folder.
- Splits each document into overlapping chunks.
- Converts every chunk into a semantic embedding.
- Stores all embeddings inside ChromaDB.

---

### Query Pipeline

When a user asks a question:

- The question is converted into an embedding.
- ChromaDB retrieves the most relevant document chunks.
- The retrieved chunks are supplied to the language model as context.
- Groq's Llama 3.3 70B generates an answer based only on the retrieved information.
- The application displays both the generated answer and the retrieved source chunks.

---

## Running the Project

### Clone the repository

```bash
git clone <repository-url>
cd RAG-internship
```

### Create a virtual environment

```bash
python -m venv ragas_env
```

### Activate the environment

Windows

```bash
ragas_env\Scripts\activate
```

Linux / macOS

```bash
source ragas_env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure the environment

Create a `.env` file.

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

### Generate the vector database

```bash
python ingest.py
```

### Launch the chatbot

```bash
streamlit run app.py
```

---

## Sample Questions

- What is Retrieval-Augmented Generation?
- What is a vector database?
- Explain semantic search.
- Explain the RAG pipeline.

---

## Learning Outcomes

This project provided practical experience with:

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Document Chunking
- Sentence Embeddings
- Vector Databases
- ChromaDB
- LangChain
- Streamlit Deployment
- Prompt Engineering
- Retrieval-based Question Answering

---

## Internship Objectives Achieved

This project fulfills the following internship objectives:

- Built a document ingestion pipeline.
- Implemented semantic chunking and embeddings.
- Stored embeddings inside ChromaDB.
- Developed a retrieval pipeline for semantic search.
- Integrated Groq's Llama 3.3 70B for answer generation.
- Built a Streamlit-based user interface.
- Successfully deployed the application for public access.

---

## Future Enhancements

The following improvements are planned as part of future iterations:

- Hybrid Search (Vector Search + BM25)
- Metadata Filtering
- Re-ranking
- PDF and DOCX support
- RAGAS Evaluation
- Multi-document collections
- Conversation memory

---

## Author

**Paul Ranjith**

Computer Science Engineering Student

Developed as part of a Product Engineering Internship on Retrieval-Augmented Generation (RAG).
