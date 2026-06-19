# query script for RAG pipeline using LangGraph. this is the main interface for asking qustions after ingest script is run

import os
import sys
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from groq import Groq
    from langgraph.graph import StateGraph
except ImportError as e:
    print(f" Missing dependency: {e}")
    print("Run: py -3.13 -m pip install chromadb langchain-community langchain-huggingface sentence-transformers groq langgraph")
    sys.exit(1)


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found in .env file.")
    sys.exit(1)


CHROMA_DB_FOLDER = "./chroma_db"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K            = 5
LLM_MODEL        = "llama-3.3-70b-versatile"

if not os.path.exists(CHROMA_DB_FOLDER):
    print(" chroma_db/ not found. Run ingest.py first.")
    sys.exit(1)

print(" Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("  Loading ChromaDB from disk...")
vectorstore = Chroma(
    persist_directory=CHROMA_DB_FOLDER,
    embedding_function=embeddings
)

groq_client = Groq(api_key=GROQ_API_KEY)
print(" Everything loaded.\n")


#  RAG State
# same pattern as rag_pipeline.py — a bag that flows through every node
class RAGState(TypedDict):
    question:        str    # user's question
    retrieved_chunks: list  # top K chunks from ChromaDB
    sources:         list   # source filenames of retrieved chunks
    answer:          str    # final generated answer
    error:           str    # error message if anything goes wrong


# node 1 to retrieve relevant chunks from ChromaDB based on the question
def retrieve_node(state: RAGState) -> RAGState:
    """
    Search ChromaDB for the top K chunks most relevant to the question.
    Uses the same HuggingFace embedding model as ingest.py.
    LangChain handles the embedding of the question automatically.
    """
    try:
        if not state["question"].strip():
            raise ValueError("Question is empty.")

        results = vectorstore.similarity_search(state["question"], k=TOP_K)

        if not results:
            raise ValueError("No relevant chunks found in ChromaDB.")

        chunks  = [doc.page_content for doc in results]
        sources = list(set(doc.metadata.get("source", "unknown") for doc in results))

        print(f"  Retrieve: {len(chunks)} chunks from {sources}")
        return {**state, "retrieved_chunks": chunks, "sources": sources, "error": ""}

    except Exception as e:
        print(f"   Retrieve failed: {e}")
        return {**state, "error": f"Retrieve error: {e}"}


# node 2 to generate an answer using groq llm based on the retrieved chunks and the question
def generate_node(state: RAGState) -> RAGState:

    if state.get("error"):
        return {**state, "answer": f"Pipeline failed: {state['error']}"}

    try:
        context = "\n\n".join(state["retrieved_chunks"])
        prompt  = (
            "You are a helpful assistant. "
            "Answer the question using ONLY the context provided below.\n"
            "If the answer is not in the context, say: "
            "'I don't have enough information to answer that.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}\n\n"
            "Answer:"
        )

        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content.strip()
        if not answer:
            raise ValueError("LLM returned empty response.")

        print(f"  Generate: answer ready")
        return {**state, "answer": answer, "error": ""}

    except Exception as e:
        print(f"  Generate failed: {e}")
        return {**state, "error": f"Generate error: {e}", "answer": ""}


# building the langgraph pipeline wiht the 2 nodes that were defined above.
# the output of the retrieve node is passed to the generate node and the final output is returned to the user.

graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.add_edge("retrieve", "generate")

graph.set_entry_point("retrieve")
graph.set_finish_point("generate")

pipeline = graph.compile()
print("✅ LangGraph pipeline compiled.\n")


# ── Ask function ──────────────────────────────────────────────────────────────
def ask(question: str) -> str:
    """Run the full RAG query pipeline for a given question."""
    print(f"\n{'─' * 60}")
    print(f"❓ Question: {question}")
    print("─" * 60)

    initial_state: RAGState = {
        "question":         question,
        "retrieved_chunks": [],
        "sources":          [],
        "answer":           "",
        "error":            "",
    }

    result = pipeline.invoke(initial_state)

    if result.get("error"):
        print(f"\n⚠️  Error: {result['error']}")
        return ""

    print(f"\n Answer:\n{result['answer']}")
    print(f"\n Sources: {', '.join(result['sources'])}")
    return result["answer"]


# loop to simulate interaction with the user. keeps asking questions until the user types quit
def interactive_loop():
    """Keep asking questions until the user types quit."""
    print("🤖 RAG Bot ready! Ask anything about AI, ML, RAG, and more.")
    print("   Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        ask(question)
        print()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Query Pipeline — Phase 2 Task 4 (LangGraph)")
    print("=" * 60 + "\n")

    interactive_loop()