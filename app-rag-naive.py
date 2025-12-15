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
CHROMA_STORAGE_PATH = "chroma_persistent_storage_naive"
COLLECTION_NAME = "neuroscience_rag_collection"
OPENROUTER_MODEL = "google/gemma-3-27b-it:free"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

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
    model_name=EMBEDDINGS_MODEL
)

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=local_ef,  # Use the local embedding function
)


# -----------------------------------------------------
# 📄 RAG STEP 1 — INDEXING PHASE
# Load → Chunk → Embed (Vectorize) → Store
# -----------------------------------------------------


# Function to Load documents from a PDF file
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


# Function to Split documents into chunks
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
    doc_id = documents[0]["id"]  # Assuming only one PDF document
    chunks_list = split_text(documents[0]["text"])
    print(f"Text split into {len(chunks_list)} chunks.")

    for i, chunk_data in enumerate(chunks_list):
        chunked_documents.append(
            {
                "id": f"{doc_id}_chunk{i+1}",
                "text": chunk_data["text"],
                "source": doc_id,  # Keep track of the source
            }
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
def query_documents(question, n_results=4):
    """
    Performs a vector search in ChromaDB.
    """
    print(f"==== Retrieving top {n_results} relevant chunks ====")
    # Chroma handles the embedding of the question using the SentenceTransformer model
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas"],  # Include text and metadata
    )

    # Flatten the documents list (it's a list of lists because query_texts is a list)
    relevant_chunks = results.get("documents", [[]])[0]

    if relevant_chunks:
        # Also print the sources for debugging/context
        sources = [
            meta.get("source", "N/A") for meta in results.get("metadatas", [[]])[0]
        ]
        print(f"Retrieved chunks from sources: {set(sources)}")

    return relevant_chunks


# -----------------------------------------------------
# 💡 RAG STEP 3 — GENERATION PHASE
# LLM uses context to answer
# -----------------------------------------------------
def generate_response(question, relevant_chunks):
    """
    Uses the retrieved chunks as context for the LLM via OpenRouter.
    """
    if not relevant_chunks:
        return "I could not find any relevant information in the provided document to answer your question."

    context = "\n\n".join(relevant_chunks)

    # System prompt guides the LLM to use the context and follow constraints
    system_prompt = (
        "You are an assistant for question-answering tasks grounded in neuroscience. "
        "Use ONLY the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know and that the information "
        "was not in the provided documents. Keep the answer concise and strictly based on the context."
    )

    # The final prompt sent to the LLM
    llm_prompt = f"Context:\n---\n{context}\n---\n\nQuestion:\n{question}"

    print(f"==== Generating response using {OPENROUTER_MODEL} via OpenRouter... ====")

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": llm_prompt},
            ],
            # Optional: Low temperature (e.g., 0.1) encourages consistent, factual answers grounded in the context.
            temperature=0.1,
            max_tokens=256,
        )

        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        return f"An error occurred during LLM generation: {e}"


# -----------------------------------------------------
# ⚙️ QUERY EXECUTION
# -----------------------------------------------------

question = "What is the primary function of the hippocampus in the brain?"
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
