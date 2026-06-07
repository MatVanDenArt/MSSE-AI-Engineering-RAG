"""
This script handles the ingestion of markdown documents into a ChromaDB vector store.

It performs the following steps:
1.  Finds all markdown files in the 'data/' directory.
2.  Splits the documents into chunks using a two-stage process:
    a. Splits by markdown headers to preserve semantic context.
    b. Splits the resulting sections into smaller, fixed-size chunks.
3.  Uses a custom REST-based embedding class to convert text chunks into vectors
    by calling the Google Gemini API. This is optimized for low-memory environments.
4.  Creates and persists the Chroma vector database in the './chroma_db' directory.

This script should be run once to build the database, or whenever the source
documents in the 'data/' directory are updated.
"""
import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

def main():
    """
    Main function to run the document ingestion pipeline.
    """
    # Load environment variables
    load_dotenv()
    
    # Path to the data folder and ChromaDB directory
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(ROOT_DIR, "data")
    persist_directory = os.path.join(ROOT_DIR, "chroma_db")
    
    # Get all markdown files in the data directory
    md_files = glob.glob(os.path.join(data_dir, "*.md"))
    
    if not md_files:
        print(f"No markdown files found in {data_dir}")
        return
        
    print(f"Found {len(md_files)} markdown files. Starting ingestion...")
    
    # Define headers to split on for MarkdownHeaderTextSplitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Define recursive character splitter for chunking text within sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    all_chunks = []
    
    # Process each markdown file
    for file_path in md_files:
        print(f"Processing {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            markdown_document = f.read()
            
        # 1. Split by Markdown headers
        md_header_splits = markdown_splitter.split_text(markdown_document)
        
        # Add source metadata to each split
        for split in md_header_splits:
            split.metadata["source"] = os.path.basename(file_path)
            
        # 2. Split further using RecursiveCharacterTextSplitter
        chunks = text_splitter.split_documents(md_header_splits)
        all_chunks.extend(chunks)
        
    print(f"Created {len(all_chunks)} chunks from {len(md_files)} files.")
    
    # Initialize the custom REST embedding model
    print("Initializing custom GeminiRESTEmbeddings...")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    embeddings = GeminiRESTEmbeddings(api_key=gemini_key)
    
    # Create and persist the Chroma vector database
    print(f"Saving vector database to {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=all_chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    print("Ingestion complete. Vector database built successfully.")

class GeminiRESTEmbeddings(Embeddings):
    """
    Custom LangChain Embeddings class that uses the Google Gemini REST API.

    This class is a memory-efficient alternative to loading a local sentence-transformer
    model, which can exceed the memory limits of free cloud hosting tiers. It sends
    HTTP requests to the Gemini API for embedding, offloading the computational work.

    It supports batching documents to improve performance.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={api_key}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents using the Gemini batch embedding endpoint.

        Args:
            texts: A list of strings (documents) to embed.

        Returns:
            A list of lists of floats, where each inner list is the embedding
            for a corresponding document.
        """
        # Gemini API supports batching up to 100 documents per request.
        batch_url = self.url.replace(":embedContent", ":batchEmbedContents")
        
        requests_payload = [
            {"model": "models/gemini-embedding-2", "content": {"parts": [{"text": text}]}}
            for text in texts
        ]
        
        response = requests.post(batch_url, json={"requests": requests_payload})
        response.raise_for_status()
        
        return [item["values"] for item in response.json()["embeddings"]]

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding of the query.
        """
        return self.embed_documents([text])[0]

if __name__ == "__main__":
    main()
