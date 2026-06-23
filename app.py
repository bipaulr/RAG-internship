# streamlit app for the RAG chatbot
# runs the query pipeline and displays answers with source chunks

import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from groq import Groq
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.stop()

# page config — must be the first streamlit call
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

CHROMA_DB_FOLDER = "./chroma_db"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K            = 5
LLM_MODEL        = "llama-3.3-70b-versatile"


# load everything once and cache it so it doesnt reload on every interaction
# st.cache_resource keeps the model and db in memory across reruns
@st.cache_resource
def load_resources():
    # reads api key from streamlit secrets when deployed
    # falls back to .env for local development
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not groq_api_key:
        st.error("GROQ_API_KEY not found. Add it to .streamlit/secrets.toml or your .env file.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if not os.path.exists(CHROMA_DB_FOLDER):
        st.error("chroma_db/ not found. Run ingest.py first.")
        st.stop()

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_FOLDER,
        embedding_function=embeddings
    )

    groq_client = Groq(api_key=groq_api_key)
    return vectorstore, groq_client


# run the rag pipeline for a single question
def ask(question: str, vectorstore, groq_client) -> dict:
    results  = vectorstore.similarity_search(question, k=TOP_K)
    chunks   = [doc.page_content for doc in results]
    sources  = list(set(doc.metadata.get("source", "unknown") for doc in results))
    context  = "\n\n".join(chunks)

    prompt = (
        "You are a helpful assistant. "
        "Answer the question using ONLY the context provided below.\n"
        "If the answer is not in the context, say: "
        "'I don't have enough information to answer that.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer":  response.choices[0].message.content.strip(),
        "chunks":  chunks,
        "sources": sources
    }


# load models and db on first run — cached after that
with st.spinner("Loading embedding model and document database... (first load only)"):
    vectorstore, groq_client = load_resources()

# ui layout
st.title("RAG Chatbot")
st.write("Ask anything about AI, Machine Learning, RAG, and related topics.")
st.write("Powered by LangGraph + ChromaDB + Groq Llama 3.3 70B")
st.divider()

# keep chat history in session state so messages persist across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "chunks" in msg:
            with st.expander("See retrieved document chunks"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    st.caption(f"Chunk {i} — {msg['sources'][i-1] if i <= len(msg['sources']) else 'unknown'}")
                    st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.divider()

# chat input at the bottom
question = st.chat_input("Ask a question about your documents...")

if question:
    # show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # generate and show answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = ask(question, vectorstore, groq_client)
                st.write(result["answer"])

                # show sources
                st.caption(f"Sources: {', '.join(result['sources'])}")

                # expandable chunk viewer
                with st.expander("See retrieved document chunks"):
                    for i, chunk in enumerate(result["chunks"], 1):
                        source = result["sources"][i-1] if i <= len(result["sources"]) else "unknown"
                        st.caption(f"Chunk {i} — {source}")
                        st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                        st.divider()

                # save to session state with chunks for history display
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result["answer"],
                    "chunks":  result["chunks"],
                    "sources": result["sources"]
                })

            except Exception as e:
                st.error(f"Error: {e}")