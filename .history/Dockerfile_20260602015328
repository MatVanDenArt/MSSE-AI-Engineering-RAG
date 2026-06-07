# Use an official Python slim image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Render will dynamically inject the PORT environment variable
# We do NOT hardcode EXPOSE or CMD here.
# The render.yaml blueprint will pass the precise startup command (uvicorn or streamlit) 
# and override the entrypoint dynamically based on which service it's spinning up.
