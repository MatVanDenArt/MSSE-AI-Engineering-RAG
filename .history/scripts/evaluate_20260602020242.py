import time
import requests
import pandas as pd
import numpy as np

def main():
    print("Connecting to the FastAPI Backend to evaluate 15 HR questions...")
    
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
    
    print(f"Starting evaluation of {len(questions)} questions using Groq API...\n")
    
    for idx, question in enumerate(questions):
        print(f"[{idx+1}/{len(questions)}] Question: {question}")
        
        start_time = time.time()
        
        try:
            # Call the REST API Backend
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={"question": question, "provider": "Groq"},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            answer = data["answer"]
            
            # Format context from sources
            context_str = "\n\n".join([doc.get("content", "") for doc in data.get("sources", [])])
            
        except Exception as e:
            answer = f"ERROR: {str(e)}"
            context_str = ""
            
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
        
        print(f"  -> Latency: {latency:.2f} seconds")
        print(f"  -> Answer: {answer[:100].replace('\n', ' ')}...\n")
        
        results.append({
            "Question": question,
            "LLM Answer": answer,
            "Retrieved Context": context_str,
            "Latency": latency
        })
        
        # Artificial sleep to avoid Groq Free Tier rate limits
        if idx < len(questions) - 1:
            time.sleep(3)
            
    import os
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(ROOT_DIR, "evaluation_results.csv")
    md_path = os.path.join(ROOT_DIR, "evaluation_results.md")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Evaluation complete! Results saved to {csv_path}")
    
    # Save to Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# HR Assistant Evaluation Results\n\n")
        f.write("This document contains the automated evaluation results of the 15 test questions.\n\n")
        
        for idx, res in enumerate(results):
            f.write(f"### Q{idx+1}: {res['Question']}\n")
            f.write(f"**Latency:** `{res['Latency']:.2f}s`\n\n")
            
            # Format the quote block safely
            formatted_answer = res['LLM Answer'].replace('\n', '\n> ')
            f.write(f"**LLM Answer:**\n> {formatted_answer}\n\n")
            
            f.write(f"<details><summary>View Retrieved Context</summary>\n\n```text\n{res['Retrieved Context']}\n```\n</details>\n\n")
            f.write("---\n\n")
    print("Results also saved to evaluation_results.md")
    
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
