import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder

load_dotenv()

app = FastAPI(title="Wood Group HR API")

# Initialize Retriever at startup (Fetch Top 15 instead of 4)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

# Initialize CrossEncoder for Re-ranking
print("Loading CrossEncoder model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("CrossEncoder loaded successfully!")

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
        # Stage 1: Base Retrieval (Top 15)
        base_docs = retriever.invoke(request.question)
        
        # Stage 2: Re-ranking
        if base_docs:
            pairs = [[request.question, doc.page_content] for doc in base_docs]
            scores = cross_encoder.predict(pairs)
            
            # Sort documents by their CrossEncoder score descending
            scored_docs = list(zip(scores, base_docs))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Take absolute best Top 4
            final_docs = [doc for score, doc in scored_docs[:4]]
        else:
            final_docs = []
        
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
        raise HTTPException(status_code=500, detail=str(e))
