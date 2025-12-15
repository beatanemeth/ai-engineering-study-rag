# 📚 RAG Learning Project: Naive vs. HyDE

> A hands-on project to understand and implement two **Retrieval-Augmented Generation** (RAG) architectures: **Naive RAG** and **HyDE (Hypothetical Document Embedding)**, using a local PDF knowledge base.

## Project Story & Motivation ✨

This project was developed to gain practical familiarity with **AI Engineering** techniques, specifically **Retrieval-Augmented Generation** (RAG).

The project began with foundational learning:

- Inspiration was drawn from [Baraa's video](https://youtu.be/nrG1iRJW1rA) and related [roadmap documentation](https://youtu.be/nrG1iRJW1rA).

- A key resource for RAG basics was a [video on RAG bases](https://www.youtube.com/watch?v=ea2W8IogX80), which provided the theoretical framework for the hands-on practice in this project.

Building on this foundational knowledge and successful _LangChain experimentation_ —[AI Engineering - Study LangChain](https://github.com/beatanemeth/ai-engineering-study-langchain) —, the project **evolved into a solution for a real-world use case**, which is detailed in the separate repository: [AI Engineering - Custom Wix Data Chat](https://github.com/beatanemeth/ai-engineering-custom-wix-data-chat).

<br></br>

## Table of Contents

1.  [Key Learning Goals](#1-key-learning-goals-)
2.  [RAG Architectures Implemented](#2-rag-architectures-implemented-)
3.  [Technical Stack](#3-technical-stack-️)
4.  [Prerequisites](#4-prerequisites-)
5.  [Getting Started](#5-getting-started-)
6.  [Resources](#6-resources-)
7.  [Alternative Installation Method](#7-alternative-installation-method)

<br></br>

## 1. Key Learning Goals 🎯

The core purpose of this project was to achieve the following objectives:

- **Fundamentals of RAG:** Understanding the core steps: Load, Chunk, Embed/Index, Retrieve, and Generate.
- **Vector Databases:** Practical use of **ChromaDB** for persistent vector storage.
- **Embedding Models:** Utilizing local **Sentence Transformers** (`all-MiniLM-L6-v2`) for vector creation.
- **LLM Integration:** Connecting an external LLM via **OpenRouter** (`google/gemma-3-27b-it:free`) for the final answer generation.

### RAG Techniques Explored

Based on the theoretical framework, the above-mentioned video provided, it was learned that there are different RAG techniques:

- **Naive RAG** : The standard baseline approach where the user query is embedded directly and used for vector search.

- **Advanced RAG** (Query Expansion): Techniques that modify the query before retrieval.

  - **Hypothetical-Document-Embedding** (HyDE): An advanced method where the LLM first generates a hypothetical answer to the query. This synthetic answer is then embedded to perform a semantically richer retrieval.

  - **Query Expansion (Multiple Queries)**: (Explored conceptually) Uses the LLM to generate additional queries to broaden the context search.

The learning process implemented two architectures in this repository: **Naive RAG** and the **HyDE RAG** (Query Expansion with generated answer).  
A **Question-Answering (QA) system** was built that answers user questions grounded **only** in a specific, local knowledge base: _The Brain Facts Book_ (PDF).

<br></br>

## 2. RAG Architectures Implemented 🧠

This repository implements two distinct RAG techniques for comparison:

| File               | Technique     | Description                                                                                                                                                                                                     |
| :----------------- | :------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app-rag-naive.py` | **Naive RAG** | The standard approach. The original user query is directly embedded and used for vector search to find relevant document chunks.                                                                                |
| `app-rag-hyde.py`  | **HyDE RAG**  | **Hypothetical Document Embedding**. The LLM first generates a _hypothetical_ answer to the query. This synthetic answer is then embedded and used for retrieval, often resulting in a superior semantic match. |

<br></br>

## 3. Technical Stack 🛠️

| Component           | Detail                                                                        | Use                                                                               |
| :------------------ | :---------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **LLM Provider**    | OpenRouter: `google/gemma-3-27b-it:free`                                      | Generates the final answer and the hypothetical answer (for HyDE).                |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2`                                      | Runs locally to create embeddings for document chunks and queries/HyDE responses. |
| **Vector DB**       | [Chroma](https://docs.trychroma.com/) (Persistent)                            | Stores and indexes the document embeddings for fast retrieval.                    |
| **PDF Loader**      | `pypdf`                                                                       | Extracts text content from the source PDF.                                        |
| **Source Data**     | [The Brain Facts Book](https://www.brainfacts.org/the-brain-facts-book) (PDF) | The sole knowledge base for grounding answers.                                    |
| **Development OS**  | Linux Mint 21.2                                                               | The system used for development.                                                  |

<br></br>

## 4. Prerequisites 📦

You must have the following installed and configured:

- **Python 3.10.12+**
  > ⚠️ **Version Note:** This project was developed and tested using **Python 3.10.12**. While most dependencies will work with newer versions (e.g., Python 3.11/3.12), it is recommended using Python 3.10 or a compatible version to ensure environmental stability.
- An **OpenRouter API Key** (Set as `OPENROUTER_API_KEY` in the `.env` file).

<br></br>

<br></br>

## 5. Getting Started 🚀

### 5.1. Download Knowledge Base

1. Download a copy of [The Brain Facts Book](https://www.brainfacts.org/the-brain-facts-book) PDF.
2. Name the file exactly as: `brain_facts_book.pdf`
3. Replace the empty `brain_facts_book.pdf` file inside the project's `/data `folder with your downloaded sample.

### 5.2. Configuration (`.env`)

1.  In the root directory of this project, rename the `.env.example` to `.env`.
2.  Populate the file with your OpenRouter API key:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-Your_OpenRouter_API_Key
```

⚠️ **Security Tip**: Never commit your `.env` file to version control.

### 5.3. Setup Python Virtual Environment

It is best practice to use a virtual environment to isolate project dependencies.

#### 1. Create the environment

Run the following command in your project directory:

```Bash
python3 -m venv .venv
```

#### 2. Activate the environment

macOS/Linux:

```Bash
source .venv/bin/activate
```

Windows (Command Prompt):

```Bash
.venv\Scripts\activate.bat
```

Windows (PowerShell):

```Bash
.venv\Scripts\Activate.ps1
```

Your command prompt will now show the environment name, like `(.venv) user@host:~/project$`, indicating that it is active.

### 5.4. Update pip

```Bash
python -m pip install --upgrade pip
```

### 5.5. Install Dependencies

With the virtual environment active, install all necessary packages from `requirements.txt`:

```Bash
pip install -r requirements.txt
```

⚠️ **Troubleshooting Note**: If the command above fails with errors related to `torch` or other complex dependencies, please use the steps outlined in the [Alternative Installation Method](#7-alternative-installation-method) section below.

### 5.6. Run the Application

You can now run either the naive or the HyDE implementation. **The indexing phase will only run the first time** and populate the `chroma_persistent_storage` folder.

```Bash
# Run the Naive RAG implementation
python3 app-rag-naive.py

#OR

# Run the HyDE RAG implementation
python3 app-rag-hyde.py
```

### 5.7. Deactivate the environment

When you are finished, exit the isolated environment:

```bash
deactivate
```

<br></br>

## 6. Resources 📚

[ChromaDB](https://docs.trychroma.com/docs/overview/introduction)

[Sentence Transformer](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)

[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

[OpenRouterAi](https://openrouter.ai/)

[AI Engineering Roadmap - By Data With Baraa](https://candle-gosling-511.notion.site/AI-Engineering-Roadmap-By-Data-With-Baraa-29734b251f12804f94a2c5ffaeee8620?p=29834b251f1280c6a6f0e6615f9d88c9&pm=s)
<br></br>

---

## 7. Alternative Installation Method

If you notice any troubles, errors during installing packages by running the standard command (`pip install -r requirements.txt`), please switch to the following sequential installation.

Following the sequence below ensures that all major components, especially the difficult `torch` package, are installed correctly and in the right order.

1. Core Utilities & Data Loading:

```Bash
pip install pypdf python-dotenv requests httpx
```

Foundational tools for networking, environment setup, and PDF processing.

2. Vector Database (Chroma):

```Bash
pip install chromadb
```

Installs the latest stable ChromaDB and its dependencies (like numpy, scipy, pydantic).

3. LLM Client:

```Bash
pip install openai
```

Client for interacting with OpenAI or similar services.

4. PyTorch (CPU-Only):

```Bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

5. NLP Models (Embeddings):

```Bash
pip install sentence-transformers transformers
```

Installs the libraries needed to download and run embedding models (which rely on the PyTorch installed in Step 4).
