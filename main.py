import streamlit as st
import fitz  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline


PDF_PATH = "Resume_Rushik (1).pdf" 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-large"  


@st.cache_resource(show_spinner=False)
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# === Step 2: Vectorize Document ===
@st.cache_resource(show_spinner=False)
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.from_documents(chunks, embeddings)

# === Step 3: Build Retrieval QA Chain ===
@st.cache_resource(show_spinner=False)
def build_chatbot(_vector_db):
    generator = pipeline(
        "text2text-generation",
        model=LLM_MODEL_NAME,
        tokenizer=LLM_MODEL_NAME,
        max_length=512,
        temperature=0.0,
    )
    llm = HuggingFacePipeline(pipeline=generator)
    return RetrievalQA.from_chain_type(llm=llm, retriever=_vector_db.as_retriever())


st.set_page_config(page_title="📘 PDF Q&A Chatbot", layout="centered")
st.title("📘 PDF Q&A Chatbot")
st.write("Ask questions based on the content of `Resume_Rushik (1).pdf`.")


with st.spinner("Reading and indexing the PDF..."):
    text = extract_text_from_pdf(PDF_PATH)
    vector_db = create_vector_store(text)
    try:
        qa_chain = build_chatbot(vector_db)
    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {e}")
        st.stop()


user_query = st.text_input("Enter your question:", "")

if user_query:
    with st.spinner("Searching..."):
        answer = qa_chain.run(user_query)
    st.markdown(f"**Answer:** {answer}")
