# 📚 RAG Learning Project: Naive vs. HyDE

> A hands-on project to understand and implement two **Retrieval-Augmented Generation** (RAG) architectures: **Naive RAG** and **HyDE (Hypothetical Document Embedding)**, using a local PDF knowledge base.

## Project Story & Motivation ✨

This project was developed to gain practical familiarity with **AI Engineering** techniques, specifically **Retrieval-Augmented Generation** (RAG).

The project began with foundational learning:

- Inspiration was drawn from [Baraa's video](https://youtu.be/nrG1iRJW1rA) and related [roadmap documentation](https://youtu.be/nrG1iRJW1rA).

- A key resource for RAG basics was a [video on RAG bases](https://www.youtube.com/watch?v=ea2W8IogX80), which provided the theoretical framework for the hands-on practice in this project.

### Key Learning Goals 🎯

The core purpose of this project was to achieve three objectives:

- **Understand RAG**: Define and comprehend the core concepts of Retrieval-Augmented Generation.

- **Basic RAG Steps**: Identify the fundamental stages of a RAG application (Load, Chunk, Embed, Retrieve, Generate).

- **Vector Databases**: Gain practical experience with storing and querying embeddings in a persistent vector database.

### RAG Techniques Explored

Based on the theoretical framework, the above-mentioned video provided, it was learned that there are different RAG techniques:

- **Naive RAG** : The standard baseline approach where the user query is embedded directly and used for vector search.

- **Advanced RAG** (Query Expansion): Techniques that modify the query before retrieval.

  - **Hypothetical-Document-Embedding** (HyDE): An advanced method where the LLM first generates a hypothetical answer to the query. This synthetic answer is then embedded to perform a semantically richer retrieval.

  - **Query Expansion (Multiple Queries)**: (Explored conceptually) Uses the LLM to generate additional queries to broaden the context search.

The learning process implemented two architectures in this repository: **Naive RAG** and the **HyDE RAG** (Query Expansion with generated answer).
<br></br>

## Table of Contents

1.  [Project Overview & Learning Goals](#1-project-overview--learning-goals-)
2.  [RAG Architectures Implemented](#2-rag-architectures-implemented-)
3.  [Technical Stack](#3-technical-stack-️)
4.  [Prerequisites](#4-prerequisites-)
5.  [Getting Started](#5-getting-started-)
    - [5.1. Configuration (.env)](#51-configuration-env)
    - [5.2. Download Knowledge Base](#52-download-knowledge-base)
    - [5.3. Setup Python Environment](#53-setup-python-virtual-environment)
    - [5.4. Install Dependencies](#54-install-dependencies)
    - [5.5. Run the Application](#55-run-the-application)
6.  [RAG Flow Details](#6-rag-flow-details-️)

<br></br>

## 1. Project Overview & Learning Goals 🎯

The primary goal of this project is to build a robust **Question-Answering (QA) system** that answers user questions grounded **only** in a specific, local knowledge base: _The Brain Facts Book_ (PDF).

**Key Learning Objectives:**

- **Fundamentals of RAG:** Understanding the core steps: Load, Chunk, Embed/Index, Retrieve, and Generate.
- **Vector Databases:** Practical use of **ChromaDB** for persistent vector storage.
- **Embedding Models:** Utilizing local **Sentence Transformers** (`all-MiniLM-L6-v2`) for vector creation.
- **LLM Integration:** Connecting an external LLM via **OpenRouter** (`google/gemma-3-27b-it:free`) for the final answer generation.

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

- **Python 3.x**
- An **OpenRouter API Key**

<br></br>

## 5. Getting Started 🚀

### 5.1. Configuration (`.env`)

1.  In the root directory of this project, rename the `.env.example` to `.env`.
2.  Populate the file with your OpenRouter API key:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-Your_OpenRouter_API_Key
```

⚠️ **Security Tip**: Never commit your `.env` file to version control.

### 5.2. Download Knowledge Base

1. Download a copy of [The Brain Facts Book](https://www.brainfacts.org/the-brain-facts-book) PDF.
2. Name the file exactly as: `brain_facts_book.pdf`
3. Place it inside the project's `/data `folder.

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

### 5.4. Install Dependencies

With the virtual environment active, install all necessary packages from `requirements.txt`:

```Bash
pip install -r requirements.txt
```

### 5.5. Run the Application

You can now run either the naive or the HyDE implementation. **The indexing phase will only run the first time** and populate the `chroma_persistent_storage` folder.

```Bash
# Run the Naive RAG implementation
python3 app-rag-naive.py

#OR

# Run the HyDE RAG implementation
python app-rag-hyde.py
```

When you are finished, exit the isolated environment:

```Bash
deactivate
```

<br></br>

## 6. RAG Flow Details ⚙️

Both applications follow the same fundamental three-step RAG cycle:

1. **Indexing**: The PDF is loaded, split into chunks, the chunks are embedded using the local Sentence Transformer model, and finally stored in the ChromaDB collection.

2. **Retrieval**: The query (or the HyDE answer) is embedded and searched against the Chroma index to find the top-k relevant chunks (default: 4).

3. **Generation**: The retrieved chunks (context) are concatenated and injected into a constrained prompt, instructing the LLM to answer the original question only using that provided context.
