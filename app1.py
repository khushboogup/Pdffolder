import streamlit as st
import fitz  # PyMuPDF (faster alternative to pdfplumber)
import uuid
import os
import hashlib
from sentence_transformers import SentenceTransformer
from supabase import create_client
from huggingface_hub import InferenceClient

# ========= CONFIG ==========
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]  # Hugging Face token

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
model = SentenceTransformer('all-MiniLM-L6-v2')
hf_client = InferenceClient(api_key=HF_TOKEN)

# ========= FUNCTIONS ==========
def hash_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def extract_and_chunk(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def embed_chunks(chunks):
    return model.encode(chunks, batch_size=16, show_progress_bar=True).tolist()

def store_to_supabase(chunks, embeddings, pdf_id):
    data = [{
        "id": f"chunk{i+1}",   # id will be chunk1, chunk2, ...
        "pdf_id": pdf_id,
        "text": chunk,
        "embedding": embedding
    } for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))]
    supabase.table("documents1").upsert(data).execute()

def retrieve_chunks(query, pdf_id, top_k=10):
    query_embedding = model.encode(query).tolist()
    response = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": top_k,
        "pdf_id_filter": pdf_id
    }).execute()
    relevant_chunk=[row["text"] for row in response.data] if response.data else []
    return relevant_chunk

def refine_with_llm(relevant_chunk, question):
    refinement_input = "\n\n---\n\n".join(relevant_chunk)
    prompt = f"""
    Refine the following extracted text chunks for clarity, conciseness, and improved readability.
    Keep the technical meaning accurate and explain any complex terms simply if needed.
    Text to refine:
    {refinement_input}
    Question:
    {question}"""
    response = hf_client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[
            {"role": "system", "content": "You are an expert technical editor and writer."},
            {"role": "user", "content": prompt}
            ],
        temperature=0.7,
        max_tokens=90  # Adjust based on expected output length
    )
    refined_text = response.choices[0].message.content
    return refined_text

# ========= STREAMLIT UI ==========
st.set_page_config(page_title="PDF Q&A Assistant")
st.title("📄 Ask Questions About Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        pdf_path = f"temp_{uuid.uuid4().hex}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        pdf_id = hash_pdf(pdf_path)

        # Avoid redundant processing
        existing = supabase.table("documents1").select("id").eq("pdf_id", pdf_id).execute()
        if existing.data:
            st.warning("⚠️ This PDF has already been processed. You can still ask questions.")
        else:
            chunks = extract_and_chunk(pdf_path)
            embeddings = embed_chunks(chunks)
            store_to_supabase(chunks, embeddings, pdf_id)
        os.remove(pdf_path)
    st.success("✅ PDF ready for Q&A.")

    question = st.text_input("Ask a question about the uploaded PDF:")
    if question:
        with st.spinner("Generating answer..."):
            results = retrieve_chunks(question, pdf_id)
            if not results:
                st.error("❌ No relevant chunks found.")
            else:
                answer = refine_with_llm(results, question)
                st.markdown("### Answer:")
                st.write(answer)
