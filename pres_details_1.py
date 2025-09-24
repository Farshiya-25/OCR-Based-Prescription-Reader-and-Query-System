# app_streamlit.py
import os
import io
import tempfile
import uuid
import time
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_path
from docstrange import DocumentExtractor
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

# ----------------------------
# CONFIG (set these in your .env)
# ----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_2")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
DOCSTRANGE_API_KEY = os.getenv("docstrange_api_key")

poppler_path = r"C:\Users\Abdul\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"

if not (DOCSTRANGE_API_KEY and PINECONE_API_KEY and GEMINI_API_KEY):
    st.error("Missing API keys. Set MISTRAL_API_KEY, PINECONE_API_KEY and GOOGLE_API_KEY in .env")
    st.stop()

# ----------------------------
# Clients
# ----------------------------
extractor = DocumentExtractor(api_key=DOCSTRANGE_API_KEY)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Pinecone (using same client style you used previously)
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "medical-data"
index = pc.Index(INDEX_NAME)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Helper functions
# ----------------------------
def run_docstrange_ocr(path: str):
    """
    Uses your provided helper which returns structured OCR results.
    We return markdown text for the first page result (typical for single images).
    """
    try:
        result = extractor.extract(path)
        markdown_content = result.extract_markdown()
        return markdown_content
    except Exception as e:
        st.warning(f"Mistral OCR failed for {path}: {e}")
        return ""

def save_to_pinecone(doc_id, text, patient="default",session_id = None):
    embedding = embedder.encode(text).tolist()
    index.upsert([
        {
            "id": doc_id,
            "values": embedding,
            "metadata": {"patient": patient,
                         "text": text,
                         "session_id": session_id}
        }
    ])


def query_pinecone_for_context(patient: str, user_query: str, session_id: str, top_k: int = 3):
    """
    Retrieval: embed the user query and ask Pinecone for the top_k matches filtered by patient name.
    Returns the list of matched metadata texts and the raw matches object.
    """
    q_emb = embedder.encode(user_query).tolist()
    results = index.query(vector=q_emb,
                          top_k=top_k,
                          include_metadata=True,
                          filter={"patient": {"$eq": patient}, "session_id": {"$eq": session_id}})
    contexts = [m["metadata"]["text"] for m in results["matches"]]
    return "\n\n".join(contexts)
    

def call_gemini_with_context(context: str, user_query: str):
    """
    Build a prompt with context and the user question, send to Gemini LLM.
    """

    prompt = f"""You are a helpful medical assistant. Use the patient record context to answer the user's question.
        Be concise and precise. If information isn't present in the context, say you don't have that information.

    Task:
    1. Correct possible OCR errors in drug names, dosages, and schedules.
    2. Interpret common prescription shorthand (e.g., "1-0-0" means morning only, 
       "0-1-1" means afternoon and night, "SOS" means as needed).
    3. Present the corrected prescription in a structured way:
         - Medicine name
         - Strength (mg/ml if mentioned)
         - Dosage schedule (expand shorthand into plain English)
         - Duration (if available)
         - Additional advice (if mentioned)
    4. If something is unclear, make a reasonable guess and mark with [?].

    Return only the cleaned prescription summary.

Context:
{context}

User question:
{user_query}

Answer:
"""
    # generate_content returns an object with .text (as used earlier)
    resp = gemini_model.generate_content(prompt)
    return resp.text.strip()

# ----------------------------
# Chat session helpers
# ----------------------------
def init_session():
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = []  # list of {"role": "user"/"assistant", "text": ...}
    if "last_patient" not in st.session_state:
        st.session_state.last_patient = None

def display_chat_history():
    for msg in st.session_state.chat_session:
        role = msg.get("role", "assistant")
        if role == "user":
            st.chat_message("user").markdown(msg["text"])
        else:
            st.chat_message("assistant").markdown(msg["text"])

def maybe_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Prescription OCR + RAG (Docstrange + Gemini)", layout="wide")
st.title("📄 Prescription Reader & Query System")

init_session()

# Single page UI with patient input + multiple file upload

uploaded_files = st.file_uploader(
    "Upload one or more files (PDF/JPG/PNG).",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

patient_name = st.text_input("Patient name:", value=st.session_state.get("last_patient", ""))

if uploaded_files and patient_name:
    processed_any = False
    st.session_state.session_id = str(uuid.uuid4())
    
    for file in uploaded_files:
        # Save uploaded file to a real temporary file on disk, so libraries (pdf2image / Mistral helper) can use a path
        suffix = os.path.splitext(file.name)[1] or ".bin"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(file.read())
            tmp.flush()
            doc_path = tmp.name
        finally:
            tmp.close()

        try:
            if file.type == "application/pdf" or doc_path.lower().endswith(".pdf"):
                # Convert each page to an image (PIL)
                pages = convert_from_path(doc_path, poppler_path=poppler_path) if poppler_path else convert_from_path(doc_path)
                for i, page in enumerate(pages, start=1):
                    # Save the page to a temporary image file required by process_image_ocr
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
                        page.save(img_tmp.name, format="JPEG")
                        page_img_path = img_tmp.name

                    # OCR via Mistral helper
                    ocr_text = run_docstrange_ocr(page_img_path)
                    extracted_text = "\n".join([line.strip() for line in ocr_text.splitlines() if line.strip()])

                    col1,col2 = st.columns([1,2])
                    with col1:
                        st.subheader("The uploaded document")
                        st.image(page, caption=f"{file.name} — page {i}", use_container_width=True)

                    with col2:
                        st.subheader(f"Extracted Text — {file.name} (page {i})")
                        st.text_area("", extracted_text, height=400)
                    # Save to Pinecone (one record per page)
                    save_to_pinecone(f"{patient_name}_{file.name}_page_{i+1}", ocr_text, patient_name, session_id=st.session_state.session_id)

                    # cleanup per-page temp image
                    try:
                        os.remove(page_img_path)
                    except Exception:
                        pass

            else:
                # Image file path available; save and call OCR
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as img_tmp:
                    img_tmp.write(open(doc_path, "rb").read())  # copy to a separate temp file if needed
                    img_tmp.flush()
                    img_path = img_tmp.name

                ocr_text = run_docstrange_ocr(img_path)
                extracted_text = "\n".join([line.strip() for line in ocr_text.splitlines() if line.strip()])


                col1,col2 = st.columns([1,2])
                with col1:
                    st.subheader("The uploaded document")
                    pil_img = Image.open(img_path).convert("RGB")
                    st.image(pil_img, caption=file.name, use_container_width=True)

                with col2:
                    st.subheader(f"Extracted Text — {file.name}")
                    #st.text_area("", extracted_text, height=400)
                    st.markdown(f"```\n{extracted_text}\n```")


                # Save to Pinecone: page_idx=1 for single-image files
                save_to_pinecone(f"{patient_name}_{file.name}", ocr_text, patient_name, session_id=st.session_state.session_id)

                # cleanup
                try:
                    os.remove(img_path)
                except Exception:
                    pass

            processed_any = True

        finally:
            # remove the uploaded temp file
            try:
                os.remove(doc_path)
            except Exception:
                pass

    if processed_any:
        st.success("All files processed and saved.")
        st.session_state.last_patient = patient_name

st.markdown("---")
st.header("Query about the prescriptions")

# Show chat history
display_chat_history()

# Chat input (pinned at bottom by Streamlit)
query = st.chat_input("Ask a question about the patient's prescriptions")

if query:
    if not patient_name:
        st.warning("Please provide a patient name before asking questions.")
    else:
        # Append user message
        st.session_state.chat_session.append({"role": "user", "text": query})
        st.chat_message("user").markdown(query)

        # Retrieve context from Pinecone
        contexts = query_pinecone_for_context(patient_name, query, st.session_state.session_id, top_k=3)
        if not contexts:
            assistant_text = "I couldn't find relevant records for this patient."
        else:
            # Join retrieved contexts (you may want to trim or format)
            combined_context = "\n\n".join(contexts)

            # Call Gemini LLM with context + user query
            try:
                assistant_text = call_gemini_with_context(combined_context, query)
            except Exception as e:
                assistant_text = f"LLM call failed: {e}\n\n (retrieved context was used below)\n\n{combined_context}"

        # Append assistant message and rerun so it shows in the chat history
        st.session_state.chat_session.append({"role": "assistant", "text": assistant_text})
        st.chat_message("assistant").markdown(assistant_text)
