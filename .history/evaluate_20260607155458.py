import os
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    print("Loading environment variables and initializing RAG pipeline...")
    load_dotenv()
    
    # 1. Initialize Vector Store Retriever
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 2. Setup Prompt
    template = """You are an HR Assistant for Wood Group. 
You must strictly use ONLY the provided context to answer the user's question. 
If the answer is not contained in the context, you must clearly state: "I can only answer about our policies based on the provided documents. I do not have information on that." 
Keep the answer concise, and properly format it with markdown lists if applicable.

Context:
{context}

Question: {question}

Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)
    
    # 3. Setup LLM
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=gemini_key, temperature=0)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # Test Questions
    questions = [
        "If I am absent from work due to sickness for 9 calendar days, what specific documentation must I provide to the Company?",
        "I work offshore. What exact phone number and option should I call to report a sickness absence?",
        "If I resign and leave the company exactly 8 months after returning from Maternity Leave, what percentage of my Company Maternity Pay (CMP) am I required to repay?",
        "Am I allowed to use any of my 'keeping in touch' (KIT) days during the two-week compulsory maternity leave period immediately after birth?",
        "What is the absolute maximum timeframe I have to complete my Paternity Leave after my child is born?",
        "If my adoption placement is unfortunately disrupted and the child is returned to the agency, how many weeks will my adoption leave and pay continue?",
        "If I am designated as the 'main adopter', what is the maximum number of paid adoption appointments I am entitled to take time off for?",
        "If my partner takes exactly 6 weeks of maternity leave, how many weeks of Shared Parental Leave (SPL) are left for us to share?",
        "How many 'shared parental leave in touch' (SPLIT) days can I agree to work without bringing my shared parental leave to an end?",
        "What is the maximum number of weeks of unpaid Ordinary Parental Leave I can take for a single child within one calendar year?",
        "Can the company legally postpone my requested Ordinary Parental Leave if I ask for it to start on the exact day my child is born?",
        "If I am summoned to attend court for Jury Service, will my time off be paid or unpaid?",
        "My home boiler broke down and I cannot work remotely. If I take a day off to deal with this emergency, will it be paid?",
        "How many days of paid leave does the company offer for the bereavement of an immediate family member?",
        "I am an independent contractor providing services to Wood Group in the UK. Do the company's Maternity and Paternity procedures apply to me?"
    ]
    
    results = []
    latencies = []
    
    print(f"Starting evaluation of {len(questions)} questions...\n")
    
    for idx, question in enumerate(questions):
        print(f"[{idx+1}/{len(questions)}] Question: {question}")
        
        start_time = time.time()
        
        # Retrieval
        docs = retriever.invoke(question)
        context_str = format_docs(docs)
        
        # Generation
        chain = (
            {"context": lambda x: context_str, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        try:
            answer = chain.invoke(question)
        except Exception as e:
            answer = f"ERROR: {str(e)}"
            
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
        
        print(f"  -> Latency: {latency:.2f} seconds")
        print(f"  -> Answer: {answer[:100]}...\n")
        
        results.append({
            "Question": question,
            "LLM Answer": answer,
            "Retrieved Context": context_str,
            "Latency": latency
        })
        
        # Sleep to avoid rate limits on free tier
        if idx < len(questions) - 1:
            time.sleep(4)
            
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation complete! Results saved to evaluation_results.csv")
    
    # Calculate Metrics
    latencies_np = np.array(latencies)
    avg_latency = np.mean(latencies_np)
    p50_latency = np.percentile(latencies_np, 50)
    p95_latency = np.percentile(latencies_np, 95)
    
    print("\n--- System Metrics ---")
    print(f"Total Questions Evaluated: {len(latencies)}")
    print(f"Average Latency: {avg_latency:.2f} seconds")
    print(f"p50 Latency:     {p50_latency:.2f} seconds")
    print(f"p95 Latency:     {p95_latency:.2f} seconds")

if __name__ == "__main__":
    main()
