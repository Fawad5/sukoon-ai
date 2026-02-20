import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- 1. CONFIG & ADVANCED UI STYLING ---
st.set_page_config(page_title="Sukoon AI", page_icon="🌿", layout="centered")

# Custom CSS for Gradient Background, Urdu Typography, and Highlighted Cards
st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }

    /* English Guidance Card */
    .english-card {
        background-color: #ffffff;
        border-left: 5px solid #66bb6a;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        font-family: 'Source Sans Pro', sans-serif;
        color: #34495e;
    }

    /* Urdu Guidance Card */
    .urdu-card {
        background-color: #f1f8e9;
        border-right: 8px solid #2e7d32;
        padding: 25px;
        border-radius: 12px;
        direction: rtl;
        text-align: right;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    /* Highlight Box for Ayah/Hadith */
    .highlight-box {
        background-color: #ffffff;
        border: 1px dashed #2e7d32;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        font-weight: bold;
        color: #1b5e20;
    }

    /* Font Styles */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 20px;
        line-height: 2.2;
    }
    .label {
        font-size: 12px;
        text-transform: uppercase;
        color: #999;
        margin-bottom: 8px;
        display: block;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALIZE MODELS ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("sukoon_index1", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    return vector_db, llm

vector_db, llm = load_resources()

# --- 3. THE UI ---
st.title("Sukoon AI (سکون)")
st.write("🌿 *Find spiritual peace and guidance.*")

user_input = st.text_input("How are you feeling today? / آپ کیسا محسوس کر رہے ہیں؟")

if user_input:
    if any(word in user_input.lower() for word in ["suicide", "hurt", "die", "khudkushi", "marna"]):
        st.error("Please reach out to Umang Helpline (Lahore): 0311-7786264 immediately.")
    else:
        with st.spinner("Seeking guidance..."):
            docs = vector_db.similarity_search(user_input, k=1)
            context = docs[0].page_content

            # C. STRICT PROMPT FOR BILINGUAL OUTPUT
            prompt = f"""
            You are Sukoon AI.
            User Input: {user_input}
            Context: {context}

            Follow this format EXACTLY:
            ENGLISH: [Gentle English response]
            AYAH: [The Arabic and Urdu text of the Hadith or Ayah from context]
            SEPARATOR
            URDU: [Gentle Urdu response]
            """
            
            response = llm.invoke(prompt).content

            try:
                # Splitting logic
                eng_section, urdu_section = response.split("SEPARATOR")
                eng_parts = eng_section.split("AYAH:")
                eng_text = eng_parts[0].replace("ENGLISH:", "").strip()
                ayah_text = eng_parts[1].strip()
                urdu_text = urdu_section.replace("URDU:", "").strip()

                # --- BEAUTIFUL RENDERING ---
                
                # 1. English Section
                st.markdown(f"""
                    <div class="english-card">
                        <span class="label">Gentle Guidance</span>
                        {eng_text}
                    </div>
                """, unsafe_allow_html=True)

                # 2. Highlighted Ayah/Hadith Box
                st.markdown(f"""
                    <div class="urdu-card" style="background-color: #fff; border-right: 8px solid #ffd700;">
                        <span class="label" style="text-align:right;">آیت / حدیث مبارکہ</span>
                        <div class="urdu-text" style="font-weight: bold; text-align: center;">{ayah_text}</div>
                    </div>
                """, unsafe_allow_html=True)

                # 3. Urdu Section
                st.markdown(f"""
                    <div class="urdu-card">
                        <span class="label">آپ کے لیے رہنمائی</span>
                        <div class="urdu-text">{urdu_text}</div>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                # Fallback display
                st.info("Guidance Found:")
                st.write(response)

# --- 4. FOOTER ---
st.divider()
st.caption("Sukoon AI provides spiritual support. For clinical emergencies, consult a professional.")
