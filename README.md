
# 📘 PDF Q\&A Chatbot

This project is a simple yet powerful chatbot application that allows users to query the contents of a PDF document using natural language. It uses open-source models and tools, ensuring full control, privacy, and customization for end users.

---

## 🚀 Features

* **PDF Text Extraction:** Uses PyMuPDF to extract text from each page of the PDF.
* **Text Splitting:** Splits long texts into manageable chunks using LangChain's `RecursiveCharacterTextSplitter`.
* **Semantic Embedding:** Converts text chunks to embeddings using HuggingFace’s `all-MiniLM-L6-v2`.
* **Vector Indexing:** Stores the embeddings using FAISS for efficient similarity search.
* **Language Model:** Uses `google/flan-t5-large` for generating responses.
* **User Interface:** Interactive frontend built with Streamlit.

---

## 🛠️ Tech Stack

| Component     | Tool/Library                         |
| ------------- | ------------------------------------ |
| PDF Reading   | PyMuPDF (`fitz`)                     |
| Chunking      | LangChain Text Splitter              |
| Embeddings    | HuggingFace SentenceTransformers     |
| Vector Store  | FAISS                                |
| LLM Inference | HuggingFace Transformers (`flan-t5`) |
| App Interface | Streamlit                            |

---

## 🧱 Project Structure

```bash
.
├── main.py                # Streamlit app logic
├── Resume_Rushik (1).pdf  # Target PDF document
├── requirements.txt       # Python dependencies
```

---

## 🧠 How It Works

1. **Load PDF** – Extracts all text from the uploaded/resident PDF.
2. **Text Processing** – Chunks the text into overlapping sections.
3. **Vectorization** – Embeds chunks into high-dimensional vectors.
4. **Build QA Chain** – Uses HuggingFace model to answer queries based on retrieved relevant chunks.
5. **User Interaction** – User types questions, and the app responds intelligently using relevant PDF content.

---

## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pdf-qa-chatbot.git
cd pdf-qa-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run main.py
```

---

## 📄 Example Questions to Ask

* "What are the key skills listed in the resume?"
* "Which projects has Rushik worked on?"
* "Mention the certifications included."

---

## ✅ Requirements

* Python 3.8+
* Streamlit
* langchain
* transformers
* sentence-transformers
* faiss-cpu
* PyMuPDF

---

## 🙋‍♂️ Author

**Rushik Dumpala**
For any feedback or suggestions, feel free to connect or raise issues.

---


## RESULTS:
![Screenshot 2025-06-05 152729](https://github.com/user-attachments/assets/b155bd75-f259-41bd-92ee-093210651f15)

