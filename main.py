import streamlit as st
import fitz
import random
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message

# --- Constants ---
PDF_PATH = "Resume_Rushik (1).pdf"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-large"

# --- Sentiment Analysis Model Setup (using a pretrained HuggingFace sentiment model) ---
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    positive_score = scores[0][1].item()
    negative_score = scores[0][0].item()
    if positive_score > 0.7:
        return "positive"
    elif negative_score > 0.7:
        return "negative"
    else:
        return "neutral"

# --- Friendly Buddy Style with mood-based variation ---
BUDDY_STYLES = {
    "positive": {
        "prefixes": ["Hey there! Loving your vibe! 😄 ", "Awesome! Here's what I found for you! ✨ ", "You're in a great mood! Let's dive in! 🤩 "],
        "suffixes": [" Hope you like it! Let me know if you want more! 😎", " That’s pretty cool, huh? Ask away! 👍", " Can't wait to hear what you ask next! 😊"],
        "jokes": ["Why don't scientists trust atoms? Because they make up everything! 😆", "I told a joke about a pencil once, but it had no point. 😂"]
    },
    "negative": {
        "prefixes": ["Hey, I’m here for you. Let’s find out together. 🤗 ", "Don’t worry, I got you covered. Here’s what I found. 💪 ", "Feeling a bit down? Let’s see if this helps. 🌈 "],
        "suffixes": [" Hope this helps brighten your day! 🌟", " I’m here if you want to ask anything else. 😊", " Let me know if you want to talk more! 🤝"],
        "jokes": ["Why did the scarecrow win an award? Because he was outstanding in his field! 🌾😂", "I know it’s tough, but here’s a smile: What’s orange and sounds like a parrot? A carrot! 🥕😄"]
    },
    "neutral": {
        "prefixes": ["Hey! Here's what I found. 🤖 ", "Alright, let's check this out! 👀 ", "Cool, here’s some info for you! 👍 "],
        "suffixes": [" Let me know if you want more details! 😊", " Hope this helps! Ask me anything else! 🙌", " Feel free to ask if you want to dig deeper! 🔍"],
        "jokes": ["Why did the computer go to the doctor? Because it had a virus! 🤒💻", "I’m reading a book on anti-gravity. It’s impossible to put down! 📚😄"]
    }
}

SHORT_CLARIFICATIONS = [
    "Hmm, I didn’t quite catch that. Mind rephrasing? 🤔",
    "That was a bit short! Can you tell me more? 😅",
    "I’m all ears! Try asking in a different way? 👂"
]

# --- Utility Functions ---
@st.cache_resource(show_spinner=False)
def extract_text_from_pdf(file_path):
    with fitz.open(file_path) as doc:
        return "".join([page.get_text() for page in doc])

@st.cache_resource(show_spinner=False)
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.from_documents(docs, embeddings)

@st.cache_resource(show_spinner=False)
def build_chatbot(_vector_db):
    pipe = pipeline(
        "text2text-generation",
        model=LLM_MODEL_NAME,
        tokenizer=LLM_MODEL_NAME,
        max_length=1024,
        temperature=0.7,  # creative + friendly
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    retriever = _vector_db.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

def get_buddy_style(user_text):
    mood = analyze_sentiment(user_text)
    style = BUDDY_STYLES.get(mood, BUDDY_STYLES["neutral"])
    return style, mood

def pick_joke(jokes_list):
    return random.choice(jokes_list)

# --- Streamlit Setup ---
st.set_page_config(page_title="🤗 Friendly Resume Buddy Plus", layout="centered")
st.title("🤗 Friendly Resume Buddy Plus")
st.caption("Chat with your resume like a real friend who gets you!")

# --- Load and Build Chatbot ---
with st.spinner("🧐 Getting to know the resume..."):
    text = extract_text_from_pdf(PDF_PATH)
    vector_db = create_vector_store(text)
    chatbot = build_chatbot(vector_db)

# --- Session State for chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mood" not in st.session_state:
    st.session_state.mood = "neutral"

# --- Input ---
user_input = st.chat_input("💬 Ask me anything!")

# --- Process Input ---
if user_input:
    with st.spinner("🤔 Thinking like your buddy..."):
        st.session_state.chat_history.append(("user", user_input))
        time.sleep(random.uniform(0.6, 1.2))  # simulate thinking delay
        
        # Update mood based on user input sentiment
        style, mood = get_buddy_style(user_input)
        st.session_state.mood = mood  # remember last mood for flow

        raw_response = chatbot.run(user_input).strip()
        word_count = len(raw_response.split())

        # Topic starters for more natural friend-speak
        input_lower = user_input.lower()
        if "achieve" in input_lower:
            starter = random.choice([
                "Oh wow, here are the achievements! 🎉 ",
                "His cool achievements include: ",
                "Let me tell you about his wins: ",
            ])
        elif "intern" in input_lower:
            starter = random.choice([
                "Internship details coming up! 👨‍💻 ",
                "Here’s what he did during his internships: ",
                "Let’s talk about those internship gigs: ",
            ])
        elif "educat" in input_lower or "study" in input_lower or "college" in input_lower:
            starter = random.choice([
                "Here’s what he studied in school 🎓: ",
                "Education-wise, here’s the scoop: ",
                "About his college and studies: ",
            ])
        elif "skill" in input_lower:
            starter = random.choice([
                "Skills alert! 🚨 He’s good at: ",
                "Check out these skills: ",
                "His skillset includes: ",
            ])
        else:
            starter = random.choice(style["prefixes"])

        suffix = random.choice(style["suffixes"])

        response = f"{starter}{raw_response}{suffix}"

        # Occasionally crack a joke (20% chance)
        if random.random() < 0.2:
            joke = pick_joke(style["jokes"])
            response += f"\n\n😂 BTW, here's a joke for you: {joke}"

        st.session_state.chat_history.append(("ai", response))
        st.session_state.chat_history.append(("meta", f"📝 Word Count: {word_count}"))

        if word_count < 8:
            st.session_state.chat_history.append(("note", random.choice(SHORT_CLARIFICATIONS)))

# --- Display Chat Messages ---
for i, (role, msg) in enumerate(st.session_state.chat_history):
    if role == "meta":
        st.markdown(f"<div style='color:gray;font-size:0.85em;'>{msg}</div>", unsafe_allow_html=True)
    elif role == "note":
        st.info(msg)
    else:
        is_user = (role == "user")
        message(msg, is_user=is_user, key=f"{role}_{i}")
