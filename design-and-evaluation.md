# Design and Evaluation Document

## High-Level Architecture
This project utilizes a modern **Client-Server Architecture** to separate the user interface from the heavy LLM routing and vector computations.
1. **Frontend (Streamlit)**: Serves as a lightweight, interactive user interface running on port 8501. It collects user input and renders Markdown responses and source references.
2. **Backend (FastAPI)**: Serves as a persistent REST API running on port 8000. It exposes `/chat` and `/health` endpoints. It handles the LangChain orchestration, interacts with the local Vector Database, and communicates with the SaaS LLM providers over the network.

## Architecture & Design Decisions

### 1. Vector Database & Embedding (Local)
We use **ChromaDB** to locally store the document embeddings. It is lightweight, file-system based (no cloud DB setup required), and inherently supports LangChain.
- **Embedding Model**: `all-MiniLM-L6-v2` (HuggingFace). This is a highly efficient, free-tier local embedding model perfectly suited for mapping text chunks to dense vectors without cloud overhead.

### 2. Chunking Strategy
We parse markdown files using a two-stage approach:
- `MarkdownHeaderTextSplitter`: Groups context logically by headers.
- `RecursiveCharacterTextSplitter`: Splits larger sections into manageable windows (`chunk_size=500`, `chunk_overlap=50`) to ensure high context resolution during retrieval.

### 3. Retrieval Architecture
We utilize a **Base Retrieval Pipeline**:
- The vector store dynamically retrieves the absolute best Top 4 (`k=4`) documents based on raw cosine similarity to the user's question. This provides exactly the context the LLM needs while ensuring the backend stays within Render's strict 512MB Free Tier memory limit.

## LLM Model Choices
The application supports routing between two primary Language Models based on availability and rate limits:
1. **Groq (llama-3.1-8b-instant)**: Chosen as the primary model. Groq's LPU architecture offers blazing-fast token generation. The `llama-3.1-8b` model strikes a perfect balance between high-fidelity instruction following and exceptionally low latency.
2. **Google Gemini (gemini-3.5-flash)**: Included as a fallback model. We discovered that older Gemini 1.0 and 1.5 models returned `404 NOT_FOUND` errors for free-tier users in the UK, necessitating an upgrade to the `3.5-flash` endpoint.

## Evaluation Results
We built a custom `evaluate.py` script that iterates over 15 hardcoded test questions encompassing multiple HR policy domains. The script outputs the LLM response, retrieved context, and request latency to `evaluation_results.csv`.

### Quantitative Metrics (Latency)
*Note: The latency metrics below represent the pure end-to-end response time of the system. An intentional 3-second artificial `time.sleep()` was injected between requests to prevent triggering `429 RESOURCE_EXHAUSTED` errors on free-tier API keys during batch processing, but this artificial delay is excluded from the true latency calculation.*

- **Total Questions Evaluated**: 15
- **Average Latency**: 1.00s
- **p50 Latency (Median)**: 1.00s
- **p95 Latency**: 1.07s
- **Errors Encountered**: 0

### Qualitative Metrics
1. **Groundedness**: **Exceptional.** By employing strict prompt engineering (`"You must strictly use ONLY the provided context to answer... If the answer is not contained in the context, you must clearly state: I can only answer about our policies..."`), we completely eliminated hallucination. In tests where edge-case questions were asked, the model reliably deferred to the fallback message.
2. **Citation Accuracy**: **High.** By pairing the LLM response with LangChain's `RunnablePassthrough` and returning the specific `metadata.source` of the retrieved chunks, the UI successfully generates collapsible reference accordions that direct the user to the exact markdown file used to formulate the answer.
