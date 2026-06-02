import streamlit as st
import requests

# Page configuration
st.set_page_config(page_title="HR Assistant", page_icon="📘", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .disclaimer-text {
        text-align: center;
        font-size: 0.8rem;
        color: #888;
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📘 HR Assistant")
    st.markdown("A specialised AI assistant trained strictly on the Wood Group UK Leave & Absence policies.")
    
    st.markdown("### CAPABILITIES")
    st.markdown("📄 Paternity & Maternity Procedures")
    st.markdown("📄 Shared Parental Leave")
    st.markdown("📄 Ordinary Parental Leave")
    st.markdown("📄 Sickness Absence")
    st.markdown("📄 Adoption Procedures")
    st.markdown("📄 General Leave of Absence")
    
    st.markdown("---")
    st.markdown("### SETTINGS")
    model_provider = st.selectbox("LLM Provider", ["Groq", "Gemini"])
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Wood Group HR Assistant. How can I help you with your UK leave or absence questions today?", "sources": None}
    ]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 View Source Documents ({len(msg['sources'])})"):
                for idx, doc in enumerate(msg["sources"]):
                    st.markdown(f"**Source {idx+1}:** `{doc.get('source', 'Unknown')}`")
                    st.markdown(f"*{doc.get('content', '')}*")
                    st.markdown("---")

import os

# Define the dynamic API base URL (defaults to localhost for local testing)
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# Render's 'hostport' property drops the schema, so we must prepend http:// if missing
# (We use http instead of https because this routes over Render's internal private network)
if not API_BASE_URL.startswith("http"):
    API_BASE_URL = f"http://{API_BASE_URL}"

API_URL = f"{API_BASE_URL}/chat"

# Handle user input
if prompt_input := st.chat_input("Ask a question about HR policies..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            try:
                # Call the REST API Backend dynamically
                response = requests.post(
                    API_URL,
                    json={"question": prompt_input, "provider": model_provider},
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()
                
                answer = data["answer"]
                sources = data["sources"]
                
                st.markdown(answer)
                
                if sources:
                    with st.expander(f"📄 View Source Documents ({len(sources)})"):
                        for idx, doc in enumerate(sources):
                            st.markdown(f"**Source {idx+1}:** `{doc.get('source', 'Unknown')}`")
                            st.markdown(f"*{doc.get('content', '')}*")
                            st.markdown("---")
                
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as e:
                st.error(f"An error occurred connecting to the backend API: {e}")

# Disclaimer at the bottom
st.markdown('<div class="disclaimer-text">Wood Group HR Assistant can make mistakes. Verify important information with your P&O representative.</div>', unsafe_allow_html=True)
