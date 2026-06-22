# evaluate script for RAG pipeline
# scores 20 questions using custom metrics instead of RAGAS
# faithfulness, answer relevancy, and context precision scored via Groq LLM

import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from groq import Groq
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    print("GROQ_API_KEY not found in .env file.")
    sys.exit(1)

print("API keys loaded.")

CHROMA_DB_FOLDER = "./chroma_db"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K            = 5
LLM_MODEL        = "llama-3.3-70b-versatile"

if not os.path.exists(CHROMA_DB_FOLDER):
    print("chroma_db/ not found. Run ingest.py first.")
    sys.exit(1)

print("Loading embedding model and ChromaDB...")
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
print("Ready.\n")


# 20 test questions with ground truth answers
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


# running the RAG pipeline for a single question
def run_pipeline(question: str) -> dict:
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


# scores a single answer using Groq as the judge LLM
# returns scores between 0 and 1 for each metric
def score_answer(question, answer, contexts, ground_truth) -> dict:
    context_text = "\n\n".join(contexts)

    scoring_prompt = f"""You are an evaluation judge for a RAG (Retrieval Augmented Generation) system.
Score the following answer on three metrics. Return ONLY a JSON object with three keys.

Question: {question}

Retrieved Context:
{context_text}

System Answer: {answer}

Ground Truth Answer: {ground_truth}

Score each metric from 0.0 to 1.0:

1. faithfulness: Is the answer only using information from the retrieved context? (1.0 = fully grounded in context, 0.0 = made up facts not in context)
2. answer_relevancy: Does the answer actually address the question asked? (1.0 = directly answers the question, 0.0 = irrelevant)
3. context_precision: Are the retrieved context chunks relevant to the question? (1.0 = all chunks are relevant, 0.0 = chunks are irrelevant)

Return ONLY this JSON, nothing else:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}}"""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": scoring_prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        # extract just the JSON part in case the model adds any text
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        scores = json.loads(raw[start:end])
        return {
            "faithfulness":      float(scores.get("faithfulness", 0)),
            "answer_relevancy":  float(scores.get("answer_relevancy", 0)),
            "context_precision": float(scores.get("context_precision", 0))
        }
    except Exception as e:
        print(f"  Scoring failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}


# run all 20 questions and collect results
print("Running 20 questions through pipeline...\n")

all_results = []

for i, item in enumerate(TEST_DATA):
    print(f"  [{i+1}/20] {item['question'][:60]}...")
    try:
        pipeline_result = run_pipeline(item["question"])
        scores = score_answer(
            question=item["question"],
            answer=pipeline_result["answer"],
            contexts=pipeline_result["contexts"],
            ground_truth=item["ground_truth"]
        )
        all_results.append({
            "question":         item["question"],
            "ground_truth":     item["ground_truth"],
            "answer":           pipeline_result["answer"],
            "sources":          pipeline_result["sources"],
            "faithfulness":     scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"],
            "context_precision":scores["context_precision"],
            "avg_score":        round((scores["faithfulness"] + scores["answer_relevancy"] + scores["context_precision"]) / 3, 4)
        })
        print(f"    faith={scores['faithfulness']:.2f}  relevancy={scores['answer_relevancy']:.2f}  precision={scores['context_precision']:.2f}")
    except Exception as e:
        print(f"  Failed: {e}")

print(f"\n{len(all_results)}/20 questions evaluated.\n")

# compute overall average scores
avg_faith    = round(sum(r["faithfulness"]      for r in all_results) / len(all_results), 4)
avg_relevancy= round(sum(r["answer_relevancy"]  for r in all_results) / len(all_results), 4)
avg_precision= round(sum(r["context_precision"] for r in all_results) / len(all_results), 4)

print("=" * 60)
print("EVALUATION RESULTS (RAGAS-style metrics, scored by Groq LLM)")
print("=" * 60)
print(f"Faithfulness:      {avg_faith}")
print(f"Answer Relevancy:  {avg_relevancy}")
print(f"Context Precision: {avg_precision}")
print("=" * 60)
print("Scores above 0.7 are good. Above 0.85 is excellent.")

# find the 3 worst performing questions
sorted_results = sorted(all_results, key=lambda x: x["avg_score"])
worst_3 = sorted_results[:3]

print("\n\n3 Failure Cases (lowest scoring questions):")
print("=" * 60)
for rank, r in enumerate(worst_3, 1):
    print(f"\nFailure #{rank}")
    print(f"  Question:          {r['question']}")
    print(f"  Answer:            {r['answer'][:200]}...")
    print(f"  Faithfulness:      {r['faithfulness']:.4f}")
    print(f"  Answer Relevancy:  {r['answer_relevancy']:.4f}")
    print(f"  Context Precision: {r['context_precision']:.4f}")
    print(f"  Avg Score:         {r['avg_score']:.4f}")

# save full results to file
with open("ragas_results.txt", "w", encoding="utf-8") as f:
    f.write("EVALUATION RESULTS\n")
    f.write("=" * 60 + "\n")
    f.write(f"Faithfulness:      {avg_faith}\n")
    f.write(f"Answer Relevancy:  {avg_relevancy}\n")
    f.write(f"Context Precision: {avg_precision}\n\n")
    f.write("INDIVIDUAL RESULTS\n")
    f.write("=" * 60 + "\n")
    for i, r in enumerate(all_results):
        f.write(f"\nQ{i+1}: {r['question']}\n")
        f.write(f"  Answer:            {r['answer'][:300]}\n")
        f.write(f"  Faithfulness:      {r['faithfulness']}\n")
        f.write(f"  Answer Relevancy:  {r['answer_relevancy']}\n")
        f.write(f"  Context Precision: {r['context_precision']}\n")
        f.write(f"  Sources:           {', '.join(r['sources'])}\n")
    f.write("\n\n3 FAILURE CASES\n")
    f.write("=" * 60 + "\n")
    for rank, r in enumerate(worst_3, 1):
        f.write(f"\nFailure #{rank}\n")
        f.write(f"  Question:  {r['question']}\n")
        f.write(f"  Answer:    {r['answer'][:300]}\n")
        f.write(f"  Avg Score: {r['avg_score']}\n")

print("\nFull results saved to ragas_results.txt")