"""
This script is a pre-flight diagnostic tool to verify environment setup.

It checks for the presence and validity of API keys for the services
this application depends on: Google Gemini and Groq.

For each service, it:
1.  Loads the API key from the .env file.
2.  Makes an authenticated API call to a basic endpoint (e.g., list models).
3.  Confirms that the required model for the application is available.
4.  Prints clear [SUCCESS], [WARNING], or [ERROR] messages to help developers
    quickly identify and resolve configuration issues.
"""
import os
import requests
from dotenv import load_dotenv

def test_gemini(api_key):
    """
    Tests the connectivity and model availability for the Google Gemini API.

    Args:
        api_key (str): The Google Gemini API key.
    """
    print("\n--- Testing Google Gemini API ---")
    if not api_key:
        print("[ERROR] NO GEMINI API KEY FOUND in .env")
        return
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("[SUCCESS] Successfully authenticated with Google Generative AI")
            models = response.json().get('models', [])
            supported_models = [m.get('name').replace('models/', '') for m in models if 'generateContent' in m.get('supportedGenerationMethods', [])]
            print(f"[SUCCESS] Found {len(supported_models)} models supporting content generation.")
            print(f"   Available samples: {', '.join(supported_models[:5])}")
            if "gemini-3.5-flash" in supported_models:
                print("[SUCCESS] Required model 'gemini-3.5-flash' is AVAILABLE.")
            else:
                print("[WARNING] 'gemini-3.5-flash' not found for this API key.")
        else:
            print(f"[ERROR] Failed to connect: HTTP {response.status_code}")
            print(response.json())
    except Exception as e:
        print(f"[ERROR] Error connecting to Gemini API: {e}")

def test_groq(api_key):
    """
    Tests the connectivity and model availability for the Groq API.

    Args:
        api_key (str): The Groq API key.
    """
    print("\n--- Testing Groq API ---")
    if not api_key:
        print("[ERROR] NO GROQ API KEY FOUND in .env")
        return
        
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print("[SUCCESS] Successfully authenticated with Groq API")
            data = response.json()
            models = data.get("data", [])
            model_ids = [m.get("id") for m in models]
            print(f"[SUCCESS] Found {len(model_ids)} available models.")
            print(f"   Available samples: {', '.join(model_ids[:5])}")
            if "llama-3.1-8b-instant" in model_ids:
                print("[SUCCESS] Required model 'llama-3.1-8b-instant' is AVAILABLE.")
            else:
                print("[WARNING] 'llama-3.1-8b-instant' not found for this API key.")
        else:
            print(f"[ERROR] Failed to connect: HTTP {response.status_code}")
            print(response.json())
    except Exception as e:
        print(f"[ERROR] Error connecting to Groq API: {e}")

if __name__ == "__main__":
    print("Loading environment variables from .env...")
    load_dotenv()
    
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    
    test_gemini(gemini_key)
    test_groq(groq_key)
    print("\nDiagnostics complete!")
