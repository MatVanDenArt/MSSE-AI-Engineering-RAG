import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def main():
    # Load environment variables
    load_dotenv()
    
    # Path to the data folder and ChromaDB directory
    data_dir = "./data"
    persist_directory = "./chroma_db"
    
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
    
    # Initialize the embedding model
    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and persist the Chroma vector database
    print(f"Saving vector database to {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=all_chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    print("Ingestion complete. Vector database built successfully.")

if __name__ == "__main__":
    main()
