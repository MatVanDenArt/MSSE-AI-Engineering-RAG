import pytest
from fastapi.testclient import TestClient
from api import app, format_docs
from langchain_core.documents import Document

client = TestClient(app)

def test_health_check():
    """Test the /health endpoint returns 200 OK and expected JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_format_docs():
    """Test the helper function format_docs correctly joins Document contents."""
    docs = [
        Document(page_content="Policy line 1", metadata={"source": "doc1.md"}),
        Document(page_content="Policy line 2", metadata={"source": "doc2.md"})
    ]
    formatted_str = format_docs(docs)
    assert formatted_str == "Policy line 1\n\nPolicy line 2"
