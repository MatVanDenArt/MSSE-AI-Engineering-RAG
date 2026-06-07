import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add the project root to the Python path to allow for direct execution of the script
# and to ensure that the 'api' module can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

client = TestClient(api.app)

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
    formatted_str = api.format_docs(docs)
    assert formatted_str == "Policy line 1\n\nPolicy line 2"

def test_chat_endpoint_success(mocker):
    """
    Test the /chat endpoint with mocked dependencies to ensure it returns a valid response.
    This test does NOT make real API calls to Groq/Gemini.
    """
    # 1. Mock the retriever's response
    mock_docs = [Document(page_content="This is a mock policy document.", metadata={"source": "mock_policy.md"})]
    mocker.patch.object(api.retriever, 'invoke', return_value=mock_docs)

    # 2. Mock the final answer from the LLM chain
    # We will mock the ChatGroq class to return a mock LLM object.
    # This mock LLM's invoke method will return a mock AIMessage, which the
    # StrOutputParser in the real chain will then convert to a string.
    mock_answer_content = "This is a mock answer based on the policy."
    mock_llm = mocker.MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=mock_answer_content)
    mocker.patch('api.ChatGroq', return_value=mock_llm)

    # 3. Make the request to the endpoint
    response = client.post("/chat", json={"question": "What is the policy?", "provider": "Groq"})

    # 4. Assert the response
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["answer"] == mock_answer_content
    assert len(response_data["sources"]) == 1
    assert response_data["sources"][0]["content"] == "This is a mock policy document."
    assert response_data["sources"][0]["source"] == "mock_policy.md"

    # Verify that our mocks were called
    api.retriever.invoke.assert_called_once_with("What is the policy?")
    # We can also assert that the LLM was created and invoked
    api.ChatGroq.assert_called_once()
    mock_llm.invoke.assert_called_once()
