# Design and evaluation document

## Business context & data strategy

### 1. Business value & user benefit
This HR Assistant is specifically designed for Wood Group's UK remote working hub. Many remote workers and contractors do not have direct access to internal corporate systems (like intranets or HR portals) where policies are usually hosted. This application democratizes access to crucial HR information, allowing remote employees to instantly query and receive accurate guidance on absence management without needing VPN access or waiting for HR tickets to be resolved.

### 2. Document selection
The documents utilized in this project are real Wood Group policies that are publicly available via the company's [remote working hub](https://www.woodgroup.com/pages/working-remotely). The corpus encompasses **7 distinct policy documents** comprising over **25,000 words** (roughly 50 pages of dense corporate text). It was intentionally restricted to a specific domain: **Absence Management** (Maternity, Paternity, Sickness, Adoption, and General Leave). This domain was selected because it represents a highly common problem area that generates a large volume of interlinked, complex policy queries. By focusing on a dense, interrelated topic, the RAG system serves as a highly optimized, specialized guide rather than a shallow generalist.

### 3. Data preprocessing pipeline (PDF to markdown)
Raw PDFs often contain formatting noise that can impact the quality of vector embeddings. To ensure accurate retrieval, a manual preprocessing pipeline was used:
1. **Noise removal:** The original PDFs were manually examined. Non-semantic noise (tables of contents, footers, headers, version control stamps, and document owner names) was stripped out, and the cleaned documents were re-saved as PDFs.
2. **Markdown conversion:** The cleaned PDFs were then converted into Markdown (`.md`) format.
3. **Structural styling:** Proper Markdown styling (like headers and bullet points) was applied. This step was important because the architecture relies on LangChain's `MarkdownHeaderTextSplitter`; explicitly defining the header structure in Markdown allows the splitter to chunk the text logically by semantic sections rather than arbitrarily cutting sentences in half.
4. **Image transcription:** Since LLMs cannot embed raw images during the text chunking process, key diagrams and visual examples in the original PDFs were manually transcribed into descriptive text. For example, a complex chart explaining "Shared Parental Pay Examples" was converted into a clean Markdown bulleted list, ensuring the RAG system could interpret the mathematical logic:

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

## High-level architecture
This project utilizes a modern **Client-Server Architecture** to separate the user interface from the heavy LLM routing and vector computations.
1. **Frontend (Streamlit)**: Serves as a lightweight, interactive user interface running on port 8501. It collects user input and renders Markdown responses and source references.
2. **Backend (FastAPI)**: Serves as a persistent REST API running on port 8000. It exposes `/chat` and `/health` endpoints. It handles the LangChain orchestration, interacts with the local Vector Database, and communicates with the SaaS LLM providers over the network.

## Architecture & design decisions

### 1. Vector database & embedding (custom REST API)
**ChromaDB** is used to store the document embeddings. It was selected because it is a lightweight, local, open-source vector database that requires no external cloud database provisioning, keeping the project entirely within free-tier capabilities without network latency for retrieval.
- **Original intent**: The initial intent was to use a local HuggingFace embedding model (`all-MiniLM-L6-v2`) and a CrossEncoder for a Two-Stage Retrieval pipeline to maximize fidelity.
- **The pivot (cloud memory optimization)**: During deployment, it was discovered that initializing the PyTorch engine (`sentence-transformers`) required to run even the smallest local HuggingFace models exceeded Render's strict 512MB Free Tier RAM limit, causing the container to crash with an `OOM (Out of Memory)` error.
- **The solution**: To guarantee deployment viability without upgrading to a paid tier, PyTorch was removed from the backend environment. A custom `GeminiRESTEmbeddings` class was built in Python to interface with Google's Gemini `models/gemini-embedding-2` REST API. This shifts the vector math to Google's cloud, dropping the backend memory footprint to ~150MB while keeping the semantic vectors high quality.

### 2. Chunking strategy
Markdown files are parsed using a two-stage approach:
- `MarkdownHeaderTextSplitter`: Groups context logically by headers.
- `RecursiveCharacterTextSplitter`: Splits larger sections into manageable windows (`chunk_size=500`, `chunk_overlap=50`) to ensure high context resolution during retrieval.

### 3. Retrieval architecture
A **Base Retrieval Pipeline** is utilized:
- The vector store dynamically retrieves the absolute best Top 4 (`k=4`) documents based on raw cosine similarity to the user's question. This provides exactly the context the LLM needs while ensuring the backend stays within Render's strict 512MB Free Tier memory limit.

### 4. Reproducibility & determinism
To satisfy strict reproducibility requirements, the system was designed to be inherently deterministic without relying on artificial random seeds:
- **Deterministic chunking:** Because the architecture relies on strict structural parsing (`MarkdownHeaderTextSplitter` and `RecursiveCharacterTextSplitter`) rather than semantic or randomized clustering algorithms, the ingestion pipeline naturally guarantees identical chunk outputs on every run.
- **Evaluation sampling:** During LLM generation and batch evaluation, the model temperature is hardcoded to `temperature=0`. This enforces "greedy decoding," forcing the LLMs to consistently select the highest-probability tokens rather than randomly sampling from a distribution. This acts as a functional equivalent to a fixed seed, ensuring consistent, reproducible responses across tests.

### 5. Prompt format & guardrails
The prompt template was engineered using strict instructional formatting to enforce the required project guardrails:
- **Anti-hallucination:** The prompt explicitly instructs: *"You must strictly use ONLY the provided context... If the answer is not contained in the context, you must clearly state: 'I can only answer about our policies...'"*. This reliably prevents the model from answering out-of-scope queries.
- **Formatting constraints:** The prompt requires the LLM to *"Keep the answer concise, and properly format it with markdown lists if applicable,"* while the LLM initializers are hard-capped at `max_tokens=500` to guarantee brevity.
- **Context injection:** The `{context}` variables are automatically injected via LangChain's `RunnablePassthrough()`, cleanly separating the retrieved knowledge from the user's `{question}`.

## LLM model choices
The application supports routing between two primary Language Models based on availability and rate limits:
1. **Groq (llama-3.1-8b-instant)**: Chosen as the primary model. Groq's architecture provides very fast token generation, and the `llama-3.1-8b` model offers a good balance between following instructions well and responding quickly.
2. **Google Gemini (gemini-3.5-flash)**: Included as a reliable fallback model. The `3.5-flash` endpoint was selected because it is a fast and capable lightweight LLM. It also offers generous free-tier rate limits, helping to ensure the system stays up and running smoothly even if the primary Groq model hits a rate limit.

### Resilient fallback routing
To prevent the application from hanging when free-tier API rate limits are exhausted, an automated fallback layer is implemented in the FastAPI backend:
- **Fail-fast initialization:** The LLM client libraries (`langchain-groq` and `langchain-google-genai`) are initialized with `max_retries=0`. This disables their default exponential backoff behavior, forcing the model to fail instantly upon hitting a rate limit rather than stalling the server for a minute.
- **Automated failover:** A `try...except` block wraps the Langchain invocation. If the primary requested model fails, the system catches the exception and routes the request to the secondary fallback model to answer the query.
- **Frontend transparency:** The backend payload returns a `used_provider` field. If a fallback occurs, the Streamlit frontend detects the change and displays a UI alert to the user, ensuring transparency about which model generated the response.

## Evaluation results
A custom `evaluate.py` script was built that iterates over 19 hardcoded test questions encompassing multiple HR policy domains. The script outputs the LLM response, retrieved context, and request latency to `evaluation_results.csv`.

### Quantitative metrics (latency)
*Note: The latency metrics below represent the pure end-to-end response time of the system. An intentional 3-second artificial `time.sleep()` was injected between requests to prevent triggering `429 RESOURCE_EXHAUSTED` errors on free-tier API keys during batch processing, but this artificial delay is excluded from the true latency calculation.*

**Compute vs. latency trade-off:** By migrating from a local PyTorch embedding model to the `GeminiRESTEmbeddings` API to solve Render's OOM crashes, an intentional ~700ms network transit delay was introduced. This slight increase in latency is a deliberate architectural trade-off made to drop backend memory usage by >300MB, guaranteeing deployment stability on free cloud tiers.

- **Total questions evaluated**: 19
- **Average latency**: 3.13s
- **p50 latency (median)**: 1.78s
- **p95 latency**: 8.95s
- **Errors encountered**: 0
- **Rate limit outliers**: Question 16 experienced an 18.15s latency spike because the free-tier API's "calls-per-minute" limit was temporarily reached. The system correctly handled the delay via exponential backoff rather than crashing, though this single outlier noticeably skews the average and p95 latency metrics higher than normal.

### Qualitative metrics
1. **Groundedness**: **100% (19/19)** By using specific prompt instructions (`"You must strictly use ONLY the provided context to answer... If the answer is not contained in the context, you must clearly state: I can only answer about our policies..."`), hallucinations were effectively avoided. 
2. **Citation accuracy**: **100% (19/19)** By pairing the LLM response with LangChain's `RunnablePassthrough` and returning the specific `metadata.source` of the retrieved chunks, the UI is able to generate collapsible reference accordions that point the user to the exact markdown file used to formulate the answer.
3. **Handling out-of-scope queries (edge cases):** The evaluation specifically tested questions where the provided context lacked the answer (e.g., asking about independent contractor eligibility or specific timeframes not covered in the text). In every instance, the model successfully refused to guess and instead triggered the safe fallback response.
4. **Complex policy logic & deductions:** The model demonstrated a strong ability to navigate multi-step HR logic. For example, when asked about Maternity Pay repayment terms for an employee resigning 8 months after returning, the model correctly cross-referenced the timescale bands and accurately determined a "50%" repayment was required. It also perfectly resolved nested logic regarding Shared Parental Leave scenarios.


#### Ablations
* **Retrieval K:** Testing `k=10` revealed that it frequently exceeded Render's 512MB memory limit and slightly diluted the LLM's context with irrelevant information, so `k=4` was selected.
* **Chunk size:** Initial experimentation with a `chunk_size` of 1000 characters demonstrated that 500 characters with a 50-character overlap provided much higher retrieval precision for specific HR policies.
* **Two-stage retrieval (re-ranking):** A CrossEncoder re-ranker was fully implemented to re-score and sort retrieved chunks. However, initializing the re-ranker model required PyTorch, which caused the free-tier cloud container to instantly crash `OOM (Out of Memory)`. Since the base retrieval (`k=4`) already achieves 100% groundedness on this focused corpus, the re-ranker was commented out in the codebase. It remains available as a plug-and-play enhancement for future paid-tier deployments with larger memory limits.

## Future enhancements
While the current architecture is highly optimized for a serverless, memory-constrained environment, several enhancements are proposed for future iterations:

1. **Multi-turn conversational memory (history-aware retrieval):** 
   Currently, the system operates as a stateless, single-turn answering engine. Integrating LangChain's `HistoryAwareRetriever` would allow the LLM to understand and resolve follow-up questions by passing the chat history back into the retrieval prompt, enabling a natural, continuous dialogue.
2. **Two-stage retrieval activation:** 
   Upgrading the deployment host to a paid tier with at least 1GB of RAM would allow the pre-built PyTorch CrossEncoder to be uncommented. This would enable the system to retrieve `k=15` chunks via fast vector search and accurately re-rank them down to the top 4 using the CrossEncoder, improving precision if the document corpus scales into the thousands.
3. **Hybrid search integration:** 
   Combining dense vector search with sparse keyword search (like BM25) would improve retrieval performance for exact-match HR policy numbers, obscure acronyms, or specific legislative acts where semantic meaning is less important than exact terminology.
4. **Agentic data access:** 
   Integrating function-calling capabilities would allow the RAG system to interface with an active SQL HR database. This would enable the assistant to answer deeply personalized questions (e.g., "How many days of annual leave do *I* currently have remaining?") by bridging static policy documents with dynamic employee records.