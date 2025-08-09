# demo.py
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Extract text from a PDF resume
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

resume_path = "3.pdf"  # Put a PDF in the same folder
resume_text = extract_text_from_pdf(resume_path)

# 2. Chunk text (simplest way)
def chunk_text(text, max_chars=500):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

chunks = chunk_text(resume_text)

# 3. Load embedding model (local, free)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_numpy=True)

# 4. Create FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# 5. Search with a sample job description
job_desc = "Data Science"
job_emb = model.encode([job_desc], convert_to_numpy=True)
faiss.normalize_L2(job_emb)
D, I = index.search(job_emb, k=3)

# 6. Print results
print("Top Matches:")
for score, idx in zip(D[0], I[0]):
    print(f"Score: {score:.3f} | Chunk: {chunks[idx][:100]}...")
