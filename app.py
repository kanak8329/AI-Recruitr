# app.py
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---- Helper functions ----
def extract_text_from_pdf(path):
    """Extract text from PDF."""
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_chars=500):
    """Split text into smaller chunks for embedding."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def build_faiss_index(chunks, model):
    """Create FAISS index from chunks."""
    embeddings = model.encode(chunks, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings

def search_resumes(job_desc, model, index, chunks, top_k=3):
    """Search for best resume chunks matching the job description."""
    job_emb = model.encode([job_desc], convert_to_numpy=True)
    faiss.normalize_L2(job_emb)
    D, I = index.search(job_emb, k=top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append((score, chunks[idx]))
    return results

# ---- Streamlit UI ----
st.set_page_config(page_title="AI Recruitr", layout="wide")
st.title("🧠 AI Recruitr — Smart Resume Matcher")

uploaded_file = st.file_uploader("Upload a PDF resume", type=["pdf"], accept_multiple_files=False)
job_description = st.text_area("Paste Job Description Here", height=150)
top_k = st.slider("Number of top matches to show", 1, 10, 3)

if st.button("Match"):
    if uploaded_file and job_description.strip():
        with open("temp_resume.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("Extracting text...")
        resume_text = extract_text_from_pdf("temp_resume.pdf")
        chunks = chunk_text(resume_text)

        st.info("Generating embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        index, _ = build_faiss_index(chunks, model)

        st.info("Searching for matches...")
        results = search_resumes(job_description, model, index, chunks, top_k)

        st.subheader("📋 Top Matches")
        for score, snippet in results:
            st.write(f"**Score:** {score:.3f}")
            st.write(snippet[:300] + "...")
            st.markdown("---")
    else:
        st.warning("Please upload a resume and enter a job description.")
