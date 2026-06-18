"""
Task 2 — Ingestion Pipeline
=============================
Loads all .txt files from documents/ folder,
splits into chunks, embeds using HuggingFace (local, free, no rate limits),
and stores in ChromaDB on disk.

Run ONCE:
    py -3.13 ingest.py

Requirements:
    py -3.13 -m pip install chromadb langchain langchain-community langchain-huggingface langchain-text-splitters sentence-transformers python-dotenv
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ── Dependency checks ─────────────────────────────────────────────────────────
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: py -3.13 -m pip install chromadb langchain langchain-community langchain-huggingface langchain-text-splitters sentence-transformers")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
DOCUMENTS_FOLDER = "documents"
CHROMA_DB_FOLDER = "./chroma_db"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50

# all-MiniLM-L6-v2:
# - runs fully locally on your machine (no API calls)
# - no rate limits, no daily quota, completely free
# - produces 384-dimensional vectors
# - fast and good quality for RAG tasks
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


# ── Step 1: Load all .txt files ───────────────────────────────────────────────
def load_documents(folder: str):
    """Load every .txt file in the folder into LangChain Document objects."""
    if not os.path.exists(folder):
        print(f"❌ Folder '{folder}' not found. Run fetch_documents.py first.")
        sys.exit(1)

    files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    if not files:
        print(f"❌ No .txt files found in '{folder}'.")
        sys.exit(1)

    print(f"\n📂 Loading {len(files)} files from {folder}/")
    all_docs = []

    for filename in sorted(files):
        path = os.path.join(folder, filename)
        try:
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)
            word_count = sum(len(doc.page_content.split()) for doc in docs)
            print(f"  ✅ {filename} ({word_count:,} words)")
        except Exception as e:
            print(f"  ❌ Failed to load {filename}: {e}")

    print(f"\n  Total documents loaded: {len(all_docs)}")
    return all_docs


# ── Step 2: Split into chunks ─────────────────────────────────────────────────
def split_documents(docs):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"\n✂️  Splitting complete:")
    print(f"  {len(docs)} documents → {len(chunks)} chunks")
    print(f"  Chunk size: ~{CHUNK_SIZE} characters with {CHUNK_OVERLAP} overlap")
    return chunks


# ── Step 3: Embed and store in ChromaDB ───────────────────────────────────────
def store_in_chromadb(chunks):
    """
    Embed all chunks using HuggingFace local model and store in ChromaDB.
    No API calls — runs entirely on your machine.
    No rate limits or daily quotas.
    """
    print(f"\n🔢 Loading local embedding model: {EMBEDDING_MODEL}")
    print(f"   (First run downloads ~90MB model — subsequent runs are instant)\n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"✅ Embedding model loaded.")
    print(f"🗄️  Embedding {len(chunks)} chunks and storing in ChromaDB...")
    print(f"   No rate limits — this will complete in one shot.\n")

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_FOLDER
        )
        print(f"\n✅ ChromaDB saved to {CHROMA_DB_FOLDER}/")
        print(f"   {len(chunks)} chunks stored and ready to query.")
        return vectorstore
    except Exception as e:
        print(f"❌ Failed to store in ChromaDB: {e}")
        sys.exit(1)


# ── Step 4: Verify it worked ──────────────────────────────────────────────────
def verify_chromadb(vectorstore):
    """Run a quick test query to confirm ChromaDB is working."""
    print(f"\n🔍 Running verification query: 'What is RAG?'")
    try:
        results = vectorstore.similarity_search("What is RAG?", k=2)
        print(f"  ✅ Verification passed!")
        print(f"  Top result from: {results[0].metadata.get('source', 'unknown')}")
        print(f"  Preview: {results[0].page_content[:150]}...")
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Ingestion Pipeline — Phase 2 Task 2")
    print("=" * 60)

    # delete old chroma_db if it exists (from the Gemini run)
    if os.path.exists(CHROMA_DB_FOLDER):
        import shutil
        print(f"\n🗑️  Deleting old chroma_db/ (switching embedding model)...")
        shutil.rmtree(CHROMA_DB_FOLDER)
        print(f"  ✅ Old database deleted.")

    docs        = load_documents(DOCUMENTS_FOLDER)
    chunks      = split_documents(docs)
    vectorstore = store_in_chromadb(chunks)
    verify_chromadb(vectorstore)

    print(f"\n{'=' * 60}")
    print("  Ingestion complete!")
    print("  Next step: run query.py to ask questions.")
    print(f"{'=' * 60}")