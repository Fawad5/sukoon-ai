import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- 1. CONFIG & SPIRITUAL THEME ---
st.set_page_config(page_title="Sukoon AI", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
    /* 1. Background: Soft spiritual gradient (Not White) */
    .stApp {
        background: linear-gradient(180deg, #f4f9f4 0%, #e8f5e9 100%);
    }

    /* 2. English Guidance Card */
    .english-card {
        background-color: rgba(255, 255, 255, 0.9);
        border-left: 5px solid #81c784;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        color: #2e3440;
    }

    /* 3. Ayah/Hadith Highlight (Gold Border) */
    .ayah-card {
        background-color: #ffffff;
        border: 2px solid #ffd700;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }

    /* 4. Urdu Guidance Card */
    .urdu-card {
        background-color: #ffffff;
        border-right: 8px solid #2e7d32;
        padding: 25px;
        border-radius: 15px;
        direction: rtl;
        text-align: right;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    /* 5. Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 22px;
        line-height: 2.2;
        color: #1b5e20;
    }
    .label {
        font-weight: bold;
        font-size: 12px;
        color: #999;
        text-transform: uppercase;
        display: block;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ENGINE ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("sukoon_index1", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    return vector_db, llm

vector_db, llm = load_resources()

# --- 3. UI ---
st.title("Sukoon AI | سکون")
st.markdown("##### *Your sanctuary for spiritual clarity.*")

user_input = st.text_input("How are you feeling? / آپ کیسا محسوس کر رہے ہیں؟", placeholder="Share your thoughts...")

if user_input:
    if any(word in user_input.lower() for word in ["suicide", "hurt", "die", "khudkushi", "marna"]):
        st.error("Please reach out to Umang Helpline (Lahore): 0311-7786264 immediately.")
    else:
        with st.spinner("Reflecting..."):
            docs = vector_db.similarity_search(user_input, k=1)
            context = docs[0].page_content

            # Enhanced prompt to ensure separation
            prompt = f"""
            User Input: {user_input}
            Spiritual Source: {context}

            Provide a response in 3 distinct parts:
            1. PART_ENG: A comforting English message.
            2. PART_AYAH: The specific Hadith or Ayah text.
            3. PART_URDU: A gentle Urdu explanation.
            
            Always include these markers (PART_ENG, PART_AYAH, PART_URDU).
            """
            
            response = llm.invoke(prompt).content

            try:
                # Advanced parsing to prevent "Guidance Found" error
                eng_msg = response.split("PART_ENG:")[1].split("PART_AYAH:")[0].strip()
                ayah_msg = response.split("PART_AYAH:")[1].split("PART_URDU:")[0].strip()
                urdu_msg = response.split("PART_URDU:")[1].strip()

                # Display English
                st.markdown(f'<div class="english-card"><span class="label">English Guidance</span>{eng_msg}</div>', unsafe_allow_html=True)

                # Display Ayah (Highlighted)
                st.markdown(f'<div class="ayah-card"><span class="label">Divine Source / آیت و حدیث</span><div class="urdu-text" style="text-align:center; font-weight:bold;">{ayah_msg}</div></div>', unsafe_allow_html=True)

                # Display Urdu
                st.markdown(f'<div class="urdu-card"><span class="label">اردو رہنمائی</span><div class="urdu-text">{urdu_msg}</div></div>', unsafe_allow_html=True)

            except:
                # If splitting still fails, just show the raw response but nicely
                st.info("Direct Guidance:")
                st.write(response)

# --- 4. FOOTER ---
st.caption("🌿 Sukoon AI is here to listen and guide.")
