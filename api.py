"""
FastAPI backend for the Wood Group HR Assistant.
Handles LLM routing, RAG orchestration via LangChain, and interactions with the local ChromaDB.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import time
import random
from langchain_chroma import Chroma
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration & API Key Validation ---
# Load environment variables and validate that the required API keys are present.
# This makes the server "fail fast" at startup if keys are missing.
load_dotenv()

class AppConfig:
    """
    Centralized configuration class that loads and validates environment variables.
    Provides fail-fast behavior if required API keys are missing.
    """
    def __init__(self):
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not self.groq_api_key:
            print("\033[93m[WARNING] GROQ_API_KEY not found. The 'Groq' provider will not be available.\033[0m")
        if not self.gemini_api_key:
            print("\033[93m[WARNING] GEMINI_API_KEY not found. The 'Gemini' provider and embeddings will not be available.\033[0m")

config = AppConfig()
# --- End Configuration ---

app = FastAPI(title="Wood Group HR API")

class GeminiRESTEmbeddings(Embeddings):
    """
    Custom LangChain Embeddings class that interfaces directly with Google's Gemini REST API.
    Designed to offload vector math to the cloud, circumventing Render's strict 512MB RAM limits
    by removing the need for a local PyTorch installation.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={api_key}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents in batches to respect Gemini's 100-document batch limit.
        Implements exponential backoff with jitter to handle rate limits (429 HTTP responses).
        """
        batch_url = self.url.replace(":embedContent", ":batchEmbedContents")
        all_embeddings = []
        # The Gemini API has a limit of 100 documents per batch request.
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            requests_payload = {
                "requests": [
                    {"model": "models/gemini-embedding-2", "content": {"parts": [{"text": text}]}}
                    for text in batch_texts
                ]
            }
            
            # --- Implement Exponential Backoff with Jitter ---
            max_retries = 3
            base_delay = 5  # Start with a 5-second delay
            for attempt in range(max_retries):
                try:
                    response = requests.post(batch_url, json=requests_payload)
                    response.raise_for_status()
                    break  # Success
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429 and attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(wait_time)
                    else:
                        raise
            # --- End Exponential Backoff ---
            
            batch_embeddings = [item["values"] for item in response.json()["embeddings"]]
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string."""
        return self.embed_documents([text])[0]

# Initialize Retriever at startup (Fetch Top 4 directly for Render Free Tier)
if not config.gemini_api_key:
    raise RuntimeError("Cannot start API: Gemini API key is missing for embeddings.")
embeddings = GeminiRESTEmbeddings(api_key=config.gemini_api_key)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ==============================================================================
# OPTIONAL: Two-Stage Retrieval (CrossEncoder Re-ranking)
# Note: This is commented out because loading a second PyTorch model exceeds 
# Render's 512MB Free Tier memory limit. If you deploy on a larger host, 
# uncomment the code below (and in the chat route) to enable hyper-accurate re-ranking.
#
# from sentence_transformers import CrossEncoder
# print("Loading CrossEncoder model...")
# cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# print("CrossEncoder loaded successfully!")
# ==============================================================================

# RAG Prompt
template = """You are an HR Assistant for Wood Group. 
You must strictly use ONLY the provided context to answer the user's question. 
If the answer is not contained in the context, you must clearly state: "I can only answer about our policies based on the provided documents. I do not have information on that." 
Keep the answer concise, and properly format it with markdown lists if applicable.

Context:
{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs) -> str:
    """Combines retrieved document chunks into a single formatted string."""
    return "\n\n".join(doc.page_content for doc in docs)

class ChatRequest(BaseModel):
    """Pydantic model defining the expected payload for the /chat endpoint."""
    question: str
    provider: str = "Groq"

@app.get("/health")
def health_check() -> dict:
    """Simple health check endpoint for Render deployment monitoring."""
    return {"status": "healthy"}

@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """
    Main chat endpoint. Receives a user question, retrieves relevant context from ChromaDB,
    constructs the LangChain prompt, and dynamically routes the request to the specified LLM.
    Implements automated failover if the primary LLM encounters an error or rate limit.
    """
    try:
        # Retrieve context directly from vector store
        # Use the modern .invoke() method for Runnable retrievers
        final_docs = retriever.invoke(request.question)
        
        # ==============================================================================
        # OPTIONAL: Two-Stage Retrieval Logic (Uncomment for larger hosts)
        # Note: If uncommented, remember to change the retriever `k` value above to 15!
        #
        # base_docs = retriever.invoke(request.question)
        # if base_docs:
        #     pairs = [[request.question, doc.page_content] for doc in base_docs]
        #     scores = cross_encoder.predict(pairs)
        #     scored_docs = list(zip(scores, base_docs))
        #     scored_docs.sort(key=lambda x: x[0], reverse=True)
        #     final_docs = [doc for score, doc in scored_docs[:4]]
        # else:
        #     final_docs = []
        # ==============================================================================
        
        # Define LLM instances with max_retries=0 to ensure they "fail fast" on rate limits
        llm_groq = None
        llm_gemini = None
        
        if config.groq_api_key:
            llm_groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=500, api_key=config.groq_api_key, max_retries=0)
        
        if config.gemini_api_key:
            llm_gemini = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=config.gemini_api_key, temperature=0, max_tokens=500, max_retries=0)

        # Determine the primary and fallback LLMs based on user preference
        if request.provider == "Groq":
            primary_llm = llm_groq
            primary_name = "Groq"
            fallback_llm = llm_gemini
            fallback_name = "Gemini"
        else:
            primary_llm = llm_gemini
            primary_name = "Gemini"
            fallback_llm = llm_groq
            fallback_name = "Groq"

        if not primary_llm and not fallback_llm:
            raise HTTPException(status_code=500, detail="No LLM providers are available on the server.")

        def execute_chain(llm):
            chain = (
                {"context": lambda x: format_docs(final_docs), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            return chain.invoke(request.question)

        used_provider = primary_name
        answer = None
        
        try:
            if not primary_llm:
                raise Exception(f"{primary_name} is not configured.")
            answer = execute_chain(primary_llm)
        except Exception as e:
            print(f"--- Fallback Triggered: {primary_name} failed. Error: {str(e)}")
            if fallback_llm:
                try:
                    answer = execute_chain(fallback_llm)
                    used_provider = fallback_name
                except Exception as fallback_error:
                    print(f"--- Fallback Failed: {fallback_name} also failed. Error: {str(fallback_error)}")
                    raise HTTPException(status_code=502, detail=f"Both primary and fallback models failed. Primary Error: {str(e)}. Fallback Error: {str(fallback_error)}")
            else:
                raise HTTPException(status_code=502, detail=f"Primary model {primary_name} failed and no fallback is configured. Error: {str(e)}")
        
        sources = [
            {"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")}
            for doc in final_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "used_provider": used_provider
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("--- EMBEDDING RATE LIMIT EXCEEDED ---")
            raise HTTPException(status_code=429, detail="The Google Gemini Embedding API is currently rate-limited (Google's Free Tier limits). Please wait 1-2 minutes and try again.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Log the full error to the console for easier debugging
        import traceback
        print("--- ERROR IN /chat ENDPOINT ---")
        traceback.print_exc()
        print("-----------------------------")
        raise HTTPException(status_code=500, detail=str(e))
