import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- 1. THE BEAUTY LAYER: CALM GRADIENT & GLASS CARDS ---
st.set_page_config(page_title="Sukoon AI", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
    /* 1. Main Background: Soft Sage/Mint Gradient */
    .stApp {
        background: linear-gradient(135deg, #f3f7f1 0%, #e1ebe2 100%);
    }

    /* 2. English Card: Modern Glass-morphism */
    .english-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        margin-bottom: 25px;
        text-align: left;
        direction: ltr;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* 3. Urdu Card: Traditional yet Clean */
    .urdu-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 30px;
        border-right: 8px solid #2e7d32;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        direction: rtl;
        text-align: right;
    }

    /* 4. Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 22px;
        line-height: 2.4;
        color: #1b5e20;
    }

    .eng-text {
        font-size: 18px;
        color: #2e3440;
        line-height: 1.7;
    }

    .label {
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 12px;
        color: #6a994e;
        margin-bottom: 10px;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ENGINE ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def load_engine():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("sukoon_index1", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    return vector_db, llm

vector_db, llm = load_engine()

# --- 3. THE INTERFACE ---
st.title("Sukoon AI | سکون")
st.markdown("##### *A sanctuary for your thoughts and spiritual growth.*")

user_input = st.text_input("What is on your mind? / آپ کیا سوچ رہے ہیں؟", placeholder="Write here...")

if user_input:
    # Emergency Check
    if any(word in user_input.lower() for word in ["suicide", "hurt", "die", "khudkushi", "marna"]):
        st.error("Please contact Umang (Lahore) immediately: 0311-7786264")
    else:
        with st.spinner("Finding tranquility..."):
            docs = vector_db.similarity_search(user_input, k=1)
            context = docs[0].page_content
            
            # We add a clearer instruction to the prompt to help the AI
            prompt = f"""
            User Input: {user_input}
            Context: {context}
            Task: Provide a comforting response. 
            You MUST follow this format:
            Write the English message, then the word SEPARATOR, then the Urdu message.
            """
            
            response = llm.invoke(prompt).content

            # --- IMPROVED SPLITTING LOGIC ---
            if "SEPARATOR" in response:
                eng_msg, urdu_msg = response.split("SEPARATOR", 1) # '1' ensures it only splits once
            elif "Urdu:" in response:
                # Backup: try splitting by 'Urdu:' if it forgot 'SEPARATOR'
                eng_msg, urdu_msg = response.split("Urdu:", 1)
            else:
                # If everything fails, show the whole response in English and a fallback in Urdu
                eng_msg = response
                urdu_msg = "آپ کی رہنمائی اوپر دی گئی ہے۔" 

            # Beautiful Display
            st.markdown(f"""
                <div class="english-card">
                    <span class="label">Gentle Guidance</span>
                    <div class="eng-text">{eng_msg.strip()}</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class="urdu-card">
                    <span class="label">روحانی سکون</span>
                    <div class="urdu-text">{urdu_msg.strip()}</div>
                </div>
            """, unsafe_allow_html=True)

# --- 4. FOOTER ---
st.divider()
st.caption("Sukoon AI: Authentically Islamic. Emotionally Grounded.")
