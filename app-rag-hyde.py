import os
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pypdf import PdfReader

# -----------------------------------------------------
# ⚙️ Configurations
# -----------------------------------------------------
PDF_FILE_PATH = os.path.join("data", "brain_facts_book.pdf")
OPENROUTER_MODEL = "google/gemma-3-27b-it:free"
CHROMA_STORAGE_PATH = "chroma_persistent_storage"
COLLECTION_NAME = "neuroscience_rag_collection"

# Load environment variables from .env file
load_dotenv()

# The 'openai' library is used for OpenRouter,
# but we use the OpenRouter API Key and base URL.
openrouter_key = os.getenv("OPENROUTER_API_KEY")

# Initialize the OpenAI-compatible client for OpenRouter
client = OpenAI(
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1",
)

# Initialize local SentenceTransformer Embedding Function
# This model will run locally on your laptop
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=local_ef
)


# -----------------------------------------------------
# 📄 RAG STEP 1 — INDEXING PHASE
# Load → Chunk → Embed (Vectorize) → Store
# -----------------------------------------------------


# Function: Load documents from a PDF file
def load_documents_from_pdf(pdf_path):
    print("==== Loading text from PDF file ====")
    documents = []

    # Check if the collection is already populated to avoid re-indexing
    if collection.count() > 0:
        print("Collection is already populated. Skipping indexing.")
        return documents

    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Extract text, handle None

        if not text:
            print("Error: Could not extract any text from the PDF.")
            return []

        # We treat the entire book as one document to be split into chunks
        documents.append({"id": os.path.basename(pdf_path), "text": text})

    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")

    return documents


# Function: Split documents into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    # Add a safety check to ensure text is long enough to chunk
    if len(text) <= chunk_size:
        return [{"text": text}]

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move back for overlap
        start = end - chunk_overlap

        # Prevent infinite loop if overlap is too large
        if start >= end:
            break

    return [{"text": chunk} for chunk in chunks]


## 1. Load: Load the PDF file
documents = load_documents_from_pdf(PDF_FILE_PATH)
print(f"Loaded {len(documents)} document(s)")

## 2. Chunk: Split the document
chunked_documents = []
if documents:
    doc_id = documents[0]["id"]
    chunks_list = split_text(documents[0]["text"])
    print(f"Text split into {len(chunks_list)} chunks.")

    for i, chunk in enumerate(chunks_list):
        chunked_documents.append(
            {"id": f"{doc_id}_chunk{i+1}", "text": chunk["text"], "source": doc_id}
        )

## 3. Embed & 4. Store: Create the Chroma Vector Store
if collection.count() == 0 and chunked_documents:
    print("==== Generating embeddings and inserting chunks into ChromaDB... ====")
    ids = [doc["id"] for doc in chunked_documents]
    texts = [doc["text"] for doc in chunked_documents]
    metadatas = [{"source": doc["source"]} for doc in chunked_documents]

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"Successfully indexed {len(chunked_documents)} chunks.")
else:
    print(
        "Skipping indexing as collection is already populated or no documents were loaded."
    )


# -----------------------------------------------------
# 🔍 RAG STEP 2 — RETRIEVAL PHASE
# Find relevant chunks
# -----------------------------------------------------


# ---- HyDE Step ----
def generate_hypothetical_answer(question):
    """
    Use the LLM to generate a hypothetical answer (HyDE)
    that will be embedded and used for retrieval.
    """
    hyde_prompt = (
        "Generate a concise hypothetical answer to the question. "
        "Do NOT say you are unsure. The answer will be used for document retrieval.\n\n"
        f"Question: {question}"
    )

    print("==== Generating HyDE hypothetical answer... ====")

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You generate hypothetical answers for retrieval.",
                },
                {"role": "user", "content": hyde_prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"HyDE generation failed, falling back to original query. Error: {e}")
        return question


def query_documents(question, n_results=4):
    """
    HyDE-enhanced vector retrieval.
    1. Generate hypothetical answer.
    2. Embed hypothetical answer.
    3. Retrieve chunks.
    """
    # --- HyDE rewrite ---
    hyde_query = generate_hypothetical_answer(question)

    print(f"==== Retrieving top {n_results} chunks using HyDE query ====")

    results = collection.query(
        query_texts=[hyde_query],
        n_results=n_results,
        include=["documents", "metadatas"],
    )

    chunks = results.get("documents", [[]])[0]
    return chunks


# -----------------------------------------------------
# 💡 RAG STEP 3 — GENERATION PHASE
# LLM uses context to answer
# -----------------------------------------------------
def generate_response(question, relevant_chunks):
    if not relevant_chunks:
        return "No relevant information found in the document."

    context = "\n\n".join(relevant_chunks)

    system_prompt = (
        "You are a neuroscience expert. "
        "Answer ONLY using the provided context. "
        "If the context does not contain the answer, say you do not know."
    )

    user_prompt = f"Context:\n---\n{context}\n---\n\nQuestion:\n{question}"

    print("==== Generating final answer using LLM... ====")

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Optional: Low temperature (e.g., 0.1) encourages consistent, factual answers grounded in the context.
            temperature=0.1,
            max_tokens=250,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error during LLM generation: {e}"


# -----------------------------------------------------
# ⚙️ QUERY EXECUTION
# -----------------------------------------------------

question = "What is the role of the hippocampus in memory?"
print("\n" + "=" * 50)
print(f"Query: {question}")
print("=" * 50 + "\n")

# Retrieval
relevant_chunks = query_documents(question)

# Generation
answer = generate_response(question, relevant_chunks)

print("\n" + "=" * 50)
print("FINAL ANSWER:")
print(answer)
print("=" * 50 + "\n")
