import os
import sys
import numpy as np
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except ImportError:
    print("groq importing failed")
    sys.exit(1

groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
if not groq_api_key:
    print("groq api not found")
    sys.exit(1)
groq_client = Groq(api_key=groq_api_key)

try:
    from google import genai
except ImportError:
    print("gemini importing failed")
    sys.exit(1)

try:
    from langgraph.graph import StateGraph
except ImportError:
    print("install langgraph")
    sys.exit(1)



api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if not api_key:
    print("gemini not ready")
    sys.exit(1)

client = genai.Client(api_key=api_key)
print("gemini ready")

# input document

DOCUMENT = """
Artificial intelligence (AI) is intelligence demonstrated by machines.
Machine learning is a subset of AI that allows systems to learn from data.
Deep learning uses neural networks with many layers to learn complex patterns.
Natural language processing (NLP) enables computers to understand human language.
Large language models like GPT and Gemini are trained on vast amounts of text data.
Retrieval Augmented Generation (RAG) combines retrieval systems with language models.
Vector databases store embeddings and allow fast similarity search.
Embeddings are numerical representations of text that capture semantic meaning.
Cosine similarity measures how similar two vectors are to each other.
LangGraph is a framework for building stateful, graph-based AI pipelines.
Python is the most widely used programming language for AI and machine learning.
Transformers changed NLP by introducing the attention mechanism in 2017.
Fine-tuning adapts a pre-trained model to a specific task or domain.
Prompt engineering is the practice of crafting inputs to get better outputs from LLMs.
AI agents can use tools, search the web, and take actions autonomously.
"""


def get_embedding(text: str) -> np.ndarray:
    """
    convert a string into a 768 dim embedding vector using gemini embed api

    para:
        text: string to be embedded

    Returns:
        np array with 768 dim
    """
    try:
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text,
        )
        return np.array(result.embeddings[0].values)

    except Exception as e:
        print(f"embedding failed Error: {e}")
        raise


# cos similarity is used to perform similarity search and assign a value based on how similar the question is to the chunk of the inpput document. higher the value more similar the q and chunk is

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Measure similarity between two embedding vectors. Returns 0.0 to 1.0."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# chunk_text is a fn that splits the document with newline char as the delimiter

def chunk_text(text: str) -> list[str]:
    """Split document into sentence-level chunks by newline."""
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


# rag state is the ds that is retained and flows through every stage of pipeline. in the start it only has thwe question and everything else is empty

class RAGState(TypedDict):
    question: str           # the user's question
    chunks: list            # list of text chunks from the document
    embeddings: list        # list of numpy arrays (one per chunk)
    retrieved_chunks: list  # top 3 most relevant chunks
    answer: str             # final generated answer
    error: str              # error message if anything goes wrong


#nodes of langgrapgh pipeline

def chunk_node(state: RAGState) -> RAGState:
    """Node 1 — Split the document into chunks."""
    try:
        chunks = chunk_text(DOCUMENT)
        if not chunks:
            raise ValueError("No chunks produced. Check your DOCUMENT text.")
        print(f"Chunk:    {len(chunks)} chunks created")
        return {**state, "chunks": chunks, "error": ""}
    except Exception as e:
        print(f"Chunk failed: {e}")
        return {**state, "error": f"Chunk error: {e}"}


def embed_node(state: RAGState) -> RAGState:
    """Node 2 — Embed every chunk using the Gemini embedding API."""
    if state.get("error"):
        return state  # skip this node if a previous node failed
    try:
        embeddings = []
        for chunk in state["chunks"]:
            emb = get_embedding(chunk)
            embeddings.append(emb)
        print(f"    Embed:    {len(embeddings)} embeddings "
              f"({len(embeddings[0])} dimensions each)")
        return {**state, "embeddings": embeddings, "error": ""}
    except Exception as e:
        print(f"    Embed failed: {e}")
        return {**state, "error": f"Embed error: {e}"}


def retrieve_node(state: RAGState) -> RAGState:
    """Node 3 — Find the top 3 chunks most relevant to the question."""
    if state.get("error"):
        return state
    try:
        # embed user questions to compare against embedded chunks of the document
        query_embedding = get_embedding(state["question"])

        # performing comparison
        scores = [
            (cosine_similarity(query_embedding, emb), state["chunks"][i])
            for i, emb in enumerate(state["embeddings"])
        ]

        # sort in desc order and taking top 3 highest similarity score chunks
        scores.sort(reverse=True)
        top_chunks = [chunk for _, chunk in scores[:3]]

        print(f"    Retrieve: top 3 chunks")
        for score, chunk in scores[:3]:
            preview = chunk[:70] + "..." if len(chunk) > 70 else chunk
            print(f"     [{score:.4f}] {preview}")

        return {**state, "retrieved_chunks": top_chunks, "error": ""}
    except Exception as e:
        print(f"    Retrieve failed: {e}")
        return {**state, "error": f"Retrieve error: {e}"}


def generate_node(state: RAGState) -> RAGState:
    """Node 4 — Generate an answer using the retrieved chunks as context."""
    if state.get("error"):
        return {**state, "answer": f"Pipeline failed: {state['error']}"}
    try:
        # combinin the retrieved chunks into context for llm
        context = "\n".join(state["retrieved_chunks"])

        # prompt construction to avoid hallucination and answer only based on the provided context
        prompt = (
            "You are a helpful assistant. "
            "Answer the question using ONLY the context provided below.\n"
            "If the answer is not in the context, say: "
            "'I don't have enough information to answer that.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}\n\n"
            "Answer:"
        )

        # llm call to generate answer pased on the provided prompt and context
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content.strip()
        if not answer:
            raise ValueError("LLM returned an empty response.")

        print(f"    Generate: answer ready")
        return {**state, "answer": answer, "error": ""}
    except Exception as e:
        print(f"    Generate failed: {e}")
        return {**state, "error": f"Generate error: {e}", "answer": ""}


#langgraph pipeline construction

graph = StateGraph(RAGState)

graph.add_node("chunk",    chunk_node)
graph.add_node("embed",    embed_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.add_edge("chunk",    "embed")
graph.add_edge("embed",    "retrieve")
graph.add_edge("retrieve", "generate")

graph.set_entry_point("chunk")
graph.set_finish_point("generate")

pipeline = graph.compile()
print("  Pipeline compiled.\n")



#fn to ask user the question and run the full pipeline
def ask(question: str) -> str:
    """Run the full RAG pipeline for a given question."""
    print(f"\n{'─' * 60}")
    print(f"  Question: {question}")
    print("─" * 60)

    initial_state: RAGState = {
        "question":        question,
        "chunks":          [],
        "embeddings":      [],
        "retrieved_chunks": [],
        "answer":          "",
        "error":           "",
    }

    result = pipeline.invoke(initial_state)

    if result.get("error"):
        print(f"\n   Error: {result['error']}")
        return ""

    print(f"\n  Answer: {result['answer']}")
    return result["answer"]


# hardcoded questions to test the pipeline

if __name__ == "__main__":
    ask("What is RAG and how does it work?")
    ask("What is LangGraph used for?")
    ask("What is the difference between deep learning and machine learning?")