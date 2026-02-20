import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- 1. CONFIG & BEAUTIFUL THEME (CSS) ---
st.set_page_config(page_title="Sukoon AI", page_icon="🌿", layout="centered")

# Custom CSS for a professional, spiritual aesthetic
st.markdown("""
    <style>
    /* Import Urdu Font */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    
    /* Main container styling */
    .stApp { background-color: #f9fbf9; }
    
    /* English Card Styling */
    .english-card {
        background-color: #ffffff;
        border-left: 5px solid #4caf50;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        font-family: 'Segoe UI', sans-serif;
        color: #2e3440;
        line-height: 1.6;
    }
    
    /* Urdu Card Styling */
    .urdu-card {
        background-color: #f1f8e9;
        border-right: 5px solid #2e7d32;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', serif;
        line-height: 2.2;
    }

    .urdu-text { font-size: 20px; color: #1b5e20; }
    .label { font-weight: bold; font-size: 14px; color: #666; margin-bottom: 10px; display: block; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALIZE RESOURCES ---
# Ensure your key is in Streamlit Cloud Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def load_sukoon_engine():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Using 'sukoon_index1' as per your previous setup
    vector_db = FAISS.load_local("sukoon_index1", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    return vector_db, llm

vector_db, llm = load_sukoon_engine()

# --- 3. THE INTERFACE ---
st.title("Sukoon AI (سکون)")
st.markdown("### Find Spiritual Peace & Guidance")

user_input = st.text_input("How are you feeling? / آپ کیسا محسوس کر رہے ہیں؟", placeholder="Type here...")

if user_input:
    # A. Emergency Check
    if any(word in user_input.lower() for word in ["suicide", "hurt", "die", "khudkushi", "marna"]):
        st.error("Please reach out to Umang Helpline (Lahore): 0311-7786264 immediately.")
    else:
        with st.spinner("Reflecting on Divine Wisdom..."):
            # B. Retrieval
            docs = vector_db.similarity_search(user_input, k=1)
            best_match = docs[0].page_content
            
            # C. Structured Prompt for Clean Separation
            prompt = f"""
            Context: {best_match}
            User Question: {user_input}
            
            Task: Provide a gentle, compassionate response as Sukoon AI.
            Rules:
            1. Start with an English section.
            2. End with an Urdu section.
            3. Use the keyword 'SEPARATOR' exactly once between them.
            """
            
            response_content = llm.invoke(prompt).content

            # D. Split and Render
            if "SEPARATOR" in response_content:
                eng_part, urdu_part = response_content.split("SEPARATOR")
            else:
                # Fallback if AI forgets the separator
                eng_part, urdu_part = response_content, "معذرت، اردو ترجمہ دستیاب نہیں ہے۔"

            # Render English Card
            st.markdown(f"""
                <div class="english-card">
                    <span class="label">SUKOON GUIDANCE</span>
                    {eng_part.strip()}
                </div>
            """, unsafe_allow_html=True)

            # Render Urdu Card
            st.markdown(f"""
                <div class="urdu-card">
                    <span class="label">روحانی رہنمائی</span>
                    <div class="urdu-text">{urdu_part.strip()}</div>
                </div>
            """, unsafe_allow_html=True)

# --- 4. FOOTER ---
st.divider()
st.caption("Sukoon AI is an AI companion. For medical or mental health crises, please contact a certified professional.")
