# evaluate script for RAG pipeline using RAGAS

import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from groq import Groq
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
    from ragas.llms import LangchainLLMWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:
    print(f" Missing dependency: {e}")
    print("Run: py -3.13 -m pip install ragas==0.4.3 datasets langchain-community langchain-huggingface langchain-google-genai sentence-transformers groq chromadb")
    sys.exit(1)

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found in .env file.")
    sys.exit(1)
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found in .env file. RAGAS needs it for scoring.")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
print(" API keys loaded.")

CHROMA_DB_FOLDER = "./chroma_db"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K            = 5
LLM_MODEL        = "llama-3.3-70b-versatile"

if not os.path.exists(CHROMA_DB_FOLDER):
    print("❌ chroma_db/ not found. Run ingest.py first.")
    sys.exit(1)

print(" Loading embedding model and ChromaDB...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = Chroma(
    persist_directory=CHROMA_DB_FOLDER,
    embedding_function=embeddings
)
groq_client = Groq(api_key=GROQ_API_KEY)
print(" Ready.\n")


# 20 test questions with ground truth answers for evaluation
TEST_DATA = [
    {"question": "What is Retrieval Augmented Generation?",
     "ground_truth": "Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with text generation, allowing language models to access external knowledge sources to produce more accurate and up-to-date responses."},
    {"question": "What is machine learning?",
     "ground_truth": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed, using statistical techniques to give computers the ability to learn from data."},
    {"question": "What is deep learning?",
     "ground_truth": "Deep learning is a subset of machine learning that uses neural networks with many layers to learn representations of data with multiple levels of abstraction."},
    {"question": "What is a transformer in deep learning?",
     "ground_truth": "A transformer is a deep learning architecture introduced in 2017 that uses self-attention mechanisms to process sequential data, and has become the foundation for large language models like GPT and BERT."},
    {"question": "What is natural language processing?",
     "ground_truth": "Natural language processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language."},
    {"question": "What is a large language model?",
     "ground_truth": "A large language model (LLM) is a type of AI model trained on massive amounts of text data that can generate, summarize, translate, and understand human language."},
    {"question": "What is a word embedding?",
     "ground_truth": "A word embedding is a numerical representation of a word in a continuous vector space where semantically similar words are mapped to nearby points."},
    {"question": "What is a vector database?",
     "ground_truth": "A vector database is a type of database designed to store and search high-dimensional vectors, enabling fast similarity search for applications like RAG and recommendation systems."},
    {"question": "What is prompt engineering?",
     "ground_truth": "Prompt engineering is the practice of designing and refining input prompts to guide AI language models to produce desired outputs more accurately and reliably."},
    {"question": "What is fine-tuning in deep learning?",
     "ground_truth": "Fine-tuning is the process of taking a pre-trained model and further training it on a smaller, task-specific dataset to adapt it to a particular application."},
    {"question": "What is a convolutional neural network?",
     "ground_truth": "A convolutional neural network (CNN) is a type of deep neural network primarily used for image recognition that uses convolutional layers to automatically detect features in images."},
    {"question": "What is reinforcement learning?",
     "ground_truth": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving rewards or penalties based on its actions."},
    {"question": "What is generative AI?",
     "ground_truth": "Generative AI refers to artificial intelligence systems that can generate new content such as text, images, audio, or video by learning patterns from training data."},
    {"question": "What is LangChain?",
     "ground_truth": "LangChain is a framework for developing applications powered by language models, providing tools to connect LLMs with external data sources, APIs, and other components."},
    {"question": "What is artificial intelligence?",
     "ground_truth": "Artificial intelligence is the simulation of human intelligence processes by machines, including learning, reasoning, problem solving, perception, and language understanding."},
    {"question": "What is the attention mechanism in transformers?",
     "ground_truth": "The attention mechanism in transformers allows the model to weigh the importance of different words in a sequence when processing each word, enabling it to capture long-range dependencies in text."},
    {"question": "What is supervised learning?",
     "ground_truth": "Supervised learning is a type of machine learning where the model is trained on labeled data, learning to map inputs to outputs based on example input-output pairs."},
    {"question": "What is unsupervised learning?",
     "ground_truth": "Unsupervised learning is a type of machine learning where the model learns patterns from unlabeled data without explicit guidance, often used for clustering and dimensionality reduction."},
    {"question": "What is transfer learning?",
     "ground_truth": "Transfer learning is a technique where a model trained on one task is reused as the starting point for a model on a different but related task, saving time and data."},
    {"question": "What is the difference between AI and machine learning?",
     "ground_truth": "Artificial intelligence is the broad field of creating machines that can perform tasks requiring human intelligence, while machine learning is a specific subset of AI that focuses on systems that learn from data."},
]


# running the RAG pipeline for a single question and returning the answer and sources
def run_pipeline(question: str) -> dict:
    """Run RAG pipeline and return answer + retrieved contexts."""
    results = vectorstore.similarity_search(question, k=TOP_K)
    chunks  = [doc.page_content for doc in results]
    sources = list(set(doc.metadata.get("source", "unknown") for doc in results))
    context = "\n\n".join(chunks)

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
        "answer":   response.choices[0].message.content.strip(),
        "contexts": chunks,
        "sources":  sources
    }


# collects answers for all 20 questions and runs RAGAS evaluation on them
print(" Running 20 questions through pipeline...\n")

questions     = []
answers       = []
contexts      = []
ground_truths = []
sources_list  = []

for i, item in enumerate(TEST_DATA):
    print(f"  [{i+1}/20] {item['question'][:60]}...")
    try:
        result = run_pipeline(item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(item["ground_truth"])
        sources_list.append(result["sources"])
    except Exception as e:
        print(f"  ❌ Failed: {e}")

print(f"\n {len(answers)}/20 questions answered.\n")


 
print(" Running RAGAS evaluation using Gemini for scoring...")
print("   This may take 1-2 minutes...\n")

# using gemni llm for scoring the answers with RAGAS
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY
)
ragas_llm = LangchainLLMWrapper(gemini_llm)

# initialise metrics with the judge LLM
faithfulness      = Faithfulness(llm=ragas_llm)
answer_relevancy  = AnswerRelevancy(llm=ragas_llm)
context_precision = ContextPrecision(llm=ragas_llm)

eval_dataset = Dataset.from_dict({
    "question":    questions,
    "answer":      answers,
    "contexts":    contexts,
    "ground_truth": ground_truths,
})

try:
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )

    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Faithfulness:      {results['faithfulness']:.4f}")
    print(f"  Answer Relevancy:  {results['answer_relevancy']:.4f}")
    print(f"  Context Precision: {results['context_precision']:.4f}")
    print("=" * 60)
    print("\n  Scores above 0.7 are good. Above 0.85 is excellent.")

    # individual results for each question
    df = results.to_pandas()
    df["question"] = questions
    df["sources"]  = [", ".join(s) for s in sources_list]

    print("\n\n Individual Results:")
    print("-" * 60)
    for i, row in df.iterrows():
        print(f"\nQ{i+1}: {row['question']}")
        print(f"  Faithfulness:      {row['faithfulness']:.4f}")
        print(f"  Answer Relevancy:  {row['answer_relevancy']:.4f}")
        print(f"  Context Precision: {row['context_precision']:.4f}")
        print(f"  Sources: {row['sources']}")

    # 3 lowest scoring questions
    df["avg_score"] = df[["faithfulness", "answer_relevancy", "context_precision"]].mean(axis=1)
    worst = df.nsmallest(3, "avg_score")

    print("\n\n❌ 3 Failure Cases (lowest scoring questions):")
    print("=" * 60)
    for rank, (_, row) in enumerate(worst.iterrows(), 1):
        q_idx = questions.index(row["question"])
        print(f"\nFailure #{rank}")
        print(f"  Question:     {row['question']}")
        print(f"  Answer:       {answers[q_idx][:200]}...")
        print(f"  Faithfulness:      {row['faithfulness']:.4f}")
        print(f"  Answer Relevancy:  {row['answer_relevancy']:.4f}")
        print(f"  Context Precision: {row['context_precision']:.4f}")

    # save full results to ragas_results.txt
    with open("ragas_results.txt", "w", encoding="utf-8") as f:
        f.write("RAGAS EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Faithfulness:      {results['faithfulness']:.4f}\n")
        f.write(f"Answer Relevancy:  {results['answer_relevancy']:.4f}\n")
        f.write(f"Context Precision: {results['context_precision']:.4f}\n\n")
        f.write("INDIVIDUAL RESULTS\n")
        f.write("=" * 60 + "\n")
        for i, row in df.iterrows():
            f.write(f"\nQ{i+1}: {row['question']}\n")
            f.write(f"  Answer:            {answers[i][:300]}\n")
            f.write(f"  Faithfulness:      {row['faithfulness']:.4f}\n")
            f.write(f"  Answer Relevancy:  {row['answer_relevancy']:.4f}\n")
            f.write(f"  Context Precision: {row['context_precision']:.4f}\n")
            f.write(f"  Sources:           {row['sources']}\n")
        f.write("\n\n3 FAILURE CASES\n")
        f.write("=" * 60 + "\n")
        for rank, (_, row) in enumerate(worst.iterrows(), 1):
            q_idx = questions.index(row["question"])
            f.write(f"\nFailure #{rank}\n")
            f.write(f"  Question: {row['question']}\n")
            f.write(f"  Answer:   {answers[q_idx][:300]}\n")
            f.write(f"  Avg Score: {row['avg_score']:.4f}\n")

    print("\n Full results saved to ragas_results.txt")

except Exception as e:
    print(f"❌ RAGAS scoring failed: {e}")
    print("\nSaving raw answers anyway...")
    with open("ragas_results.txt", "w", encoding="utf-8") as f:
        f.write("RAW PIPELINE ANSWERS\n")
        f.write("=" * 60 + "\n")
        for i, q in enumerate(questions):
            f.write(f"\nQ{i+1}: {q}\n")
            f.write(f"Answer: {answers[i]}\n")
            f.write(f"Sources: {', '.join(sources_list[i])}\n")
            f.write("-" * 40 + "\n")
    print(" Raw answers saved to ragas_results.txt")