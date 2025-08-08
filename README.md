# AI-Recruitr
AI Recruit
Parse resumes (PDF/DOCX), split into chunks, embed them, store embeddings in a FAISS index, and at query-time embed the job description and retrieve the top candidates.

Use LangChain to glue embeddings + LLM explanations (short match reasons).

Key libraries (official docs): FAISS for vector search, LangChain for embeddings & glue code, OpenAI / sentence-transformers for embeddings, and Streamlit for the UI.
