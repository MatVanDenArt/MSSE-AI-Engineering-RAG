"""
A simple utility script to list all available Google Gemini models.

This script authenticates with the Gemini API using the key from the .env file
and prints the names of all models that support the 'generateContent' method,
which are the models suitable for use in the RAG chain.
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
