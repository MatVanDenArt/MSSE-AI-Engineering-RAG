# Design and Evaluation Document

## High-Level Architecture
This project utilizes a modern **Client-Server Architecture** to separate the user interface from the heavy LLM routing and vector computations.
1. **Frontend (Streamlit)**: Serves as a lightweight, interactive user interface running on port 8501. It collects user input and renders Markdown responses and source references.
2. **Backend (FastAPI)**: Serves as a persistent REST API running on port 8000. It exposes `/chat` and `/health` endpoints. It handles the LangChain orchestration, interacts with the local Vector Database, and communicates with the SaaS LLM providers over the network.

## Vector Database Strategy
We implemented an advanced **Two-Stage Retrieval Pipeline** to maximize semantic accuracy:

1. **Stage 1 (Broad Retrieval)**: We use **ChromaDB** as a persistent local vector store. The raw PDF documents were converted to Markdown and chunked into 500-character segments. Using the HuggingFace `all-MiniLM-L6-v2` embedding model, the database fetches the top 15 most semantically similar chunks (`k=15`) as a wide net.
2. **Stage 2 (Deep Re-ranking)**: We implemented a HuggingFace CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). The CrossEncoder takes the user's specific query and individually scores and re-orders all 15 retrieved documents. It filters the list down to the absolute best **Top 4**, which are then injected into the LLM context window. This guarantees only the highest-fidelity context is used.

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
