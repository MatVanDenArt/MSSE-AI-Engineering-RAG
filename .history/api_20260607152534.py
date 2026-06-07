import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import time
from langchain_chroma import Chroma
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="Wood Group HR API")

class GeminiRESTEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={api_key}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_url = self.url.replace(":embedContent", ":batchEmbedContents")
        all_embeddings = []
        # The Gemini API has a limit of 100 documents per batch request.
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            requests_payload = [
                {"model": "models/gemini-embedding-2", "content": {"parts": [{"text": text}]}}
                for text in batch_texts
            ]
            
            response = requests.post(batch_url, json={"requests": requests_payload})
            response.raise_for_status()
            
            batch_embeddings = [item["values"] for item in response.json()["embeddings"]]
            all_embeddings.extend(batch_embeddings)

            # Add a small delay to respect the API's rate limits (e.g., 60 RPM)
            if i + batch_size < len(texts):
                time.sleep(1)
            
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Initialize Retriever at startup (Fetch Top 4 directly for Render Free Tier)
gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
embeddings = GeminiRESTEmbeddings(api_key=gemini_key)
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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class ChatRequest(BaseModel):
    question: str
    provider: str = "Groq"

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # Retrieve context directly from vector store
        # Use get_relevant_documents for compatibility with older LangChain versions
        # where retrievers are not yet 'Runnable' with an .invoke() method.
        final_docs = retriever.get_relevant_documents(request.question)
        
        # ==============================================================================
        # OPTIONAL: Two-Stage Retrieval Logic (Uncomment for larger hosts)
        # Note: If uncommented, remember to change the retriever `k` value above to 15!
        #
        # base_docs = retriever.get_relevant_documents(request.question)
        # if base_docs:
        #     pairs = [[request.question, doc.page_content] for doc in base_docs]
        #     scores = cross_encoder.predict(pairs)
        #     scored_docs = list(zip(scores, base_docs))
        #     scored_docs.sort(key=lambda x: x[0], reverse=True)
        #     final_docs = [doc for score, doc in scored_docs[:4]]
        # else:
        #     final_docs = []
        # ==============================================================================
        
        if request.provider == "Groq":
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        else:
            gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=gemini_key, temperature=0)
            
        chain = (
            {"context": lambda x: format_docs(final_docs), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(request.question)
        
        sources = [
            {"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")}
            for doc in final_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        # Log the full error to the console for easier debugging
        import traceback
        print("--- ERROR IN /chat ENDPOINT ---")
        traceback.print_exc()
        print("-----------------------------")
        raise HTTPException(status_code=500, detail=str(e))
