# OCR-Based-Prescription-Reader-and-Query-System

An AI-powered system that reads handwritten prescriptions, extracts structured information (patient name, drug names, dosages, instructions), and allows users to query the prescription data through a conversational interface.

## Workflow

**File Upload**

- User uploads prescription files (PDF, JPG, JPEG, PNG).

- Patient name is entered.

- Press Enter to process.

**OCR Processing**

- If PDF → convert to images (page by page).

- If Image → directly used.

- Mistral OCR extracts text from each page/image.

**Display Results**

- Left panel → Original prescription image.

- Right panel → Extracted OCR text.

**Data Storage (Pinecone)**

- Extracted text saved with:

    - patient_name

    - session_id (so queries apply only to current upload).

**Chat Querying**

- User types questions in chat box.

- Query is embedded → Pinecone retrieves relevant prescription text (filtered by patient/session).

- Retrieved text + user query → sent to Gemini LLM.

- LLM generates a human-friendly response.

## Features

📄 **OCR Extraction** – Reads prescriptions from images or PDFs.

📦 **Vector Storage** – Saves extracted text in Pinecone with patient/session filters.

💬 **Chat Query Interface** – Users can ask questions like:

- “What is the patient’s name?”

- “Which medications are prescribed?”

- “What is the dosage for Augmentin?”

✅ **Session-based Retrieval** – Queries only from the currently uploaded file, not older ones.

🧠 **AI Post-Processing** – Uses LLMs to improve readability and normalize drug/ dosage text.

## Tech Stack

**Frontend/UI** – Streamlit

**OCR** – Docstrange OCR API (researched alternatives: Mistral OCR, Google Vision)

**Vector DB** – Pinecone

**LLM** – Gemini API

**Language** – Python

## Installation

1. Clone the repository
```
git clone https://github.com/Farshiya-25/OCR-Based-Prescription-Reader-and-Query-System
```

2. Create virtual environment & install dependencies
```
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

pip install -r requirements.txt
```

3. Set up environment variables
Create a .env file in the root directory:
```
DOCSTRANGE_API_KEY=your_docstrange_key
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_pinecone_env
```
## Usage

Run the Streamlit app:
```
streamlit run pres_details.py
```

Steps:

- Upload a prescription image or PDF.

- Enter patient name and press Enter.

- OCR extracts and displays the prescription text.

- Ask questions in the chat box about the current prescription.

## Example Queries

- “What is the patient’s name?”

- “List all prescribed medicines.”

- “What is the dosage for Amoxicillin?”

- “How long should the patient take Pan 40mg?”

## Research Notes

**Compared OCR platforms**: Docstrange OCR (~90% accuracy on clean handwriting), Mistral OCR (~85%), and Google Vision (~70%).

**Final choice**: Docstrange OCR, due to balance of accuracy and easy integration.

LLM correction helps refine OCR mistakes, e.g., “IANFONG” → “PAN 40mg”.
