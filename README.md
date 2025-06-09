
# 🤗 Friendly PDF Buddy Plus

An interactive chatbot built with Streamlit that lets you chat naturally with your PDF resume. The bot extracts and indexes the resume text, understands your questions, analyzes your sentiment, and responds with mood-aware, friendly replies — including jokes!

---

## Features

- Extract text from PDF resumes using PyMuPDF (`fitz`)
- Split and embed text chunks with HuggingFace embeddings (`all-MiniLM-L6-v2`)
- Semantic search with FAISS vector store
- Conversational Q&A powered by `google/flan-t5-large` language model
- Sentiment analysis using DistilBERT to detect positive, negative, or neutral mood
- Mood-aware responses with personalized greetings, jokes, and small talk
- Maintains chat history for a natural conversation flow
- Simple, clean chat UI built with Streamlit and `streamlit_chat`

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

### How It Works

1. **PDF Text Extraction**  
   The app opens the PDF resume using PyMuPDF (`fitz`) and extracts all the text from each page.

2. **Text Chunking and Embedding**  
   The extracted text is split into smaller overlapping chunks for better context handling using LangChain's `RecursiveCharacterTextSplitter`. Each chunk is then converted into vector embeddings using the HuggingFace model `all-MiniLM-L6-v2` to capture semantic meaning.

3. **Vector Store Creation**  
   These embeddings are stored in a FAISS vector database for efficient semantic search and retrieval.

4. **Conversational Retrieval Chain**  
   When you ask a question, the chatbot retrieves relevant text chunks from the vector store and uses the language model `google/flan-t5-large` to generate a coherent answer based on the context and conversation history.

5. **Sentiment Analysis and Mood Detection**  
   The user's input is analyzed by a DistilBERT-based sentiment classifier to detect if the mood is positive, negative, or neutral.

6. **Mood-Aware Response Generation**  
   Depending on the detected sentiment, the chatbot personalizes its replies with mood-appropriate greetings, encouragement, and jokes to create a friendly, engaging experience.

7. **Conversation Memory**  
   The chatbot maintains a conversation buffer to remember the dialogue history, allowing more context-aware responses throughout the session.

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



---

## ✅ Requirements

-Python 3.8+
-streamlit
-pymupdf (fitz)
-langchain
-transformers
-sentence-transformers
-faiss-cpu
-torch
-streamlit_chat


---

## 🙋‍♂️ Author

**Rushik Dumpala**
For any feedback or suggestions, feel free to connect or raise issues.

---


## RESULTS:


![Screenshot 2025-06-09 152744](https://github.com/user-attachments/assets/7becac65-cf8a-40a8-91cc-255f18670f2f)


