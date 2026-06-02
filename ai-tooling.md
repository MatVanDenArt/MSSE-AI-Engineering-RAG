# AI Tooling & Prompting Documentation

This project was built using an advanced Pair-Programming workflow in collaboration with an AI Coding Assistant (Antigravity/Gemini). The AI assistant played a critical role in rapidly scaffolding the architecture, debugging complex API limitations, and refactoring the monolithic codebase into a production-ready system.

## How the AI Assistant Was Utilized

1. **Debugging Regional API Constraints:**
   During the initial development phase, we encountered persistent `404 NOT_FOUND` errors when attempting to use the default `gemini-pro` and `gemini-1.5-flash` models. The AI assistant helped diagnose that these legacy endpoints are often disabled for UK-based Free Tier keys. The assistant suggested a pivot to the newer `gemini-3.5-flash` model and proposed implementing `Groq` (`llama-3.1-8b-instant`) as an ultra-fast, highly reliable fallback.
2. **Evaluation Framework Generation:**
   The AI assistant wrote the `evaluate.py` script to programmatically iterate over our 15 hardcoded test questions. It identified that making 15 rapid API calls would trigger `429 RESOURCE_EXHAUSTED` rate limits on our free-tier keys, and proactively injected exception handling and pacing delays (`time.sleep`) to ensure the evaluation loop succeeded.
3. **Architectural Refactoring:**
   The AI assistant architected the decoupling of the application from a single Streamlit monolith into a pure REST API (FastAPI) and a frontend client (Streamlit) to strictly satisfy the grading rubric.
4. **CI/CD Automation:**
   The AI assistant drafted the automated `pytest` suite and the `.github/workflows/ci.yml` file to ensure the codebase possessed proper continuous integration.

## Key Prompts Used

Below are examples of the high-level prompts issued to the AI assistant to drive the project development:

### Prompt 1: Building the Evaluation Loop
> *"I need to evaluate this for the Quantic rubric. Please create a script named evaluate.py. Create a hardcoded list of these exact 15 test questions: [...]. Loop through these questions, calculate the latency of the response, and output the question, the LLM's answer, the retrieved context, and the latency to evaluation_results.csv."*

### Prompt 2: Switching Models
> *"Let's swap the gemini to groq and increase the time limit a little bit as well to handle the rate limits."*

### Prompt 3: Architectural Refactor
> *"Let's fix the project prompt and grading rubric divergences starting with the REST endpoints implementation using the Streamlit simply as the frontend caller."*

### Prompt 4: CI/CD Implementation
> *"I would like to move to CD/CI pipeline divergence from the project requirements and create required workflow. What actions/tests should I ideally include?"*
