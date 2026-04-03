import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

st.title("RAG system for PDFs")

# Check status
try:
    status = requests.get(f"{API_BASE_URL}/status").json()
    ready = status['status'] == 'ready'
    st.success(f"Ready: {ready} | Docs: {status['documents_count']} | Chunks: {status['total_chunks']}")
except:
    st.error("Backend not running")
    ready = False

# Upload
files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if st.button("Upload") and files:
    files_data = [('files', (f.name, f.getvalue(), 'application/pdf')) for f in files]
    with st.spinner("Processing..."):
        result = requests.post(f"{API_BASE_URL}/upload", files=files_data)
        if result.status_code == 200:
            st.success("Upload complete!")
            st.rerun()
        else:
            st.error("Upload failed")

# Query
if ready:
    question = st.text_input("Ask a question:")
    top_k = 3
    if st.button("Ask") and question:
        with st.spinner("Searching..."):
            result = requests.post(f"{API_BASE_URL}/query", json={"question": question, "top_k": top_k})
            if result.status_code == 200:
                data = result.json()
                st.write("**Answer:**", data['answer'])
                for i, source in enumerate(data['sources'], 1):
                    with st.expander(f"Source {i}"):
                        st.write(source['text'])
            else:
                st.error("Query failed")