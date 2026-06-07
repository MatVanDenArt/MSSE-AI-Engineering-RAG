# Design and Evaluation Document

## Business Context & Data Strategy

### 1. Business Value & User Benefit
This HR Assistant is specifically designed for Wood Group's UK remote working hub. Many remote workers and contractors do not have direct access to internal corporate systems (like intranets or HR portals) where policies are usually hosted. This application democratizes access to crucial HR information, allowing remote employees to instantly query and receive accurate guidance on absence management without needing VPN access or waiting for HR tickets to be resolved.

### 2. Document Selection
The document corpus was intentionally restricted to a specific domain: **Absence Management** (Maternity, Paternity, Sickness, Adoption, and General Leave). This domain was selected because it represents a highly common problem area that generates a large volume of interlinked, complex policy queries. By focusing on a dense, interrelated topic, the RAG system serves as a highly optimized, specialized guide rather than a shallow generalist.

### 3. Data Preprocessing Pipeline (PDF to Markdown)
Raw PDFs often contain layout noise that severely degrades the quality of vector embeddings. To ensure maximum retrieval fidelity, a rigorous manual preprocessing pipeline was implemented:
1. **Noise Removal:** The original PDFs were manually examined. Non-semantic noise (tables of contents, footers, headers, version control stamps, and document owner names) was stripped out, and the cleaned documents were re-saved as PDFs.
2. **Markdown Conversion:** The cleaned PDFs were then converted into Markdown (`.md`) format.
3. **Structural Styling:** Proper Markdown styling (headers, bullet points) was rigorously ensured. This step was critical because the architecture relies on LangChain's `MarkdownHeaderTextSplitter`; explicitly defining the header structure in Markdown allows the splitter to chunk the text logically by semantic sections rather than arbitrarily cutting sentences in half.
4. **Image Transcription:** Because LLMs cannot embed raw images during text chunking, critical diagrams and visual examples in the original PDFs were manually transcribed into descriptive text. For example, a complex chart explaining "Shared Parental Pay Examples" was converted into a clean Markdown bulleted list, ensuring the RAG system could perfectly interpret the mathematical logic:

   *Original Image from PDF:*  
   ![Shared Parental Pay Examples Chart](./docs/shared_parental_pay_chart.jpg)

   *Converted Markdown Text:*
   ```markdown
   The amount of Shared Parental Pay (ShPP) available depends on when maternity pay ends (based on a 39-week total statutory period):
   
   *   **Example 1:** If maternity pay ends after **6 weeks**, there are up to **33 weeks** of ShPP available.
   *   **Example 2:** If maternity pay ends after **26 weeks**, there are up to **13 weeks** of ShPP available.
   *   **Example 3:** If maternity pay ends after **39 weeks**, there is **0 weeks** of ShPP available.
   ```

---

## High-Level Architecture
This project utilizes a modern **Client-Server Architecture** to separate the user interface from the heavy LLM routing and vector computations.
1. **Frontend (Streamlit)**: Serves as a lightweight, interactive user interface running on port 8501. It collects user input and renders Markdown responses and source references.
2. **Backend (FastAPI)**: Serves as a persistent REST API running on port 8000. It exposes `/chat` and `/health` endpoints. It handles the LangChain orchestration, interacts with the local Vector Database, and communicates with the SaaS LLM providers over the network.

## Architecture & Design Decisions

### 1. Vector Database & Embedding (Custom REST API)
**ChromaDB** is used to store the document embeddings. 
- **Original Intent**: The initial intent was to use a local HuggingFace embedding model (`all-MiniLM-L6-v2`) and a CrossEncoder for a Two-Stage Retrieval pipeline to maximize fidelity.
- **The Pivot (Cloud Memory Optimization)**: During deployment, it was discovered that initializing the core PyTorch engine (`sentence-transformers`) required to run even the smallest local HuggingFace models exceeded Render's strict 512MB Free Tier RAM limit, causing the container to instantly crash `OOM (Out of Memory)`.
- **The Solution**: To guarantee deployment viability without upgrading to a paid tier, PyTorch was completely eradicated from the backend environment. A custom `GeminiRESTEmbeddings` class was developed in Python that manually interfaces with Google's Gemini `models/gemini-embedding-2` REST API. This offloads all vector math to Google's cloud, dropping the backend memory footprint to ~150MB while maintaining high-quality semantic vectors.

### 2. Chunking Strategy
Markdown files are parsed using a two-stage approach:
- `MarkdownHeaderTextSplitter`: Groups context logically by headers.
- `RecursiveCharacterTextSplitter`: Splits larger sections into manageable windows (`chunk_size=500`, `chunk_overlap=50`) to ensure high context resolution during retrieval.

### 3. Retrieval Architecture
A **Base Retrieval Pipeline** is utilized:
- The vector store dynamically retrieves the absolute best Top 4 (`k=4`) documents based on raw cosine similarity to the user's question. This provides exactly the context the LLM needs while ensuring the backend stays within Render's strict 512MB Free Tier memory limit.

## LLM Model Choices
The application supports routing between two primary Language Models based on availability and rate limits:
1. **Groq (llama-3.1-8b-instant)**: Chosen as the primary model. Groq's LPU architecture offers blazing-fast token generation. The `llama-3.1-8b` model strikes a perfect balance between high-fidelity instruction following and exceptionally low latency.
2. **Google Gemini (gemini-3.5-flash)**: Included as a fallback model. It was discovered that older Gemini 1.0 and 1.5 models returned `404 NOT_FOUND` errors for free-tier users in the UK, necessitating an upgrade to the `3.5-flash` endpoint.

## Evaluation Results
A custom `evaluate.py` script was built that iterates over 19 hardcoded test questions encompassing multiple HR policy domains. The script outputs the LLM response, retrieved context, and request latency to `evaluation_results.csv`.

### Quantitative Metrics (Latency)
*Note: The latency metrics below represent the pure end-to-end response time of the system. An intentional 3-second artificial `time.sleep()` was injected between requests to prevent triggering `429 RESOURCE_EXHAUSTED` errors on free-tier API keys during batch processing, but this artificial delay is excluded from the true latency calculation.*

**Compute vs. Latency Trade-off:** By migrating from a local PyTorch embedding model to the `GeminiRESTEmbeddings` API to solve Render's OOM crashes, an intentional ~500ms network transit delay was introduced. This slight increase in latency is a deliberate architectural trade-off made to drop backend memory usage by >300MB, guaranteeing deployment stability on free cloud tiers.

- **Total Questions Evaluated**: 19
- **Average Latency**: 3.13s
- **p50 Latency (Median)**: 1.78s
- **p95 Latency**: 8.95s
- **Errors Encountered**: 0

### Qualitative Metrics
1. **Groundedness**: **100% (19/19)** By employing strict prompt engineering (`"You must strictly use ONLY the provided context to answer... If the answer is not contained in the context, you must clearly state: I can only answer about our policies..."`), hallucination was completely eliminated. In tests where edge-case questions were asked, the model reliably deferred to the fallback message.
2. **Citation Accuracy**: **100% (19/19)** By pairing the LLM response with LangChain's `RunnablePassthrough` and returning the specific `metadata.source` of the retrieved chunks, the UI successfully generates collapsible reference accordions that direct the user to the exact markdown file used to formulate the answer.


#### Ablations
* **Retrieval K:** Testing `k=10` revealed that it frequently exceeded Render's 512MB memory limit and slightly diluted the LLM's context with irrelevant information, so `k=4` was selected.
* **Chunk Size:** Initial experimentation with a `chunk_size` of 1000 characters demonstrated that 500 characters with a 50-character overlap provided much higher retrieval precision for specific HR policies.