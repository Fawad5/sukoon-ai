import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- 1. CONFIG ---
st.set_page_config(page_title="Sukoon AI", page_icon="🌿", layout="centered")

# --- 2. THEME STATE ---
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# --- 3. DYNAMIC STYLING ---
if st.session_state.dark_mode:
    bg_color = "#121212"
    title_color = "#4caf50"
    text_color = "#e0e0e0"  # For English results
    subtext_color = "#aaaaaa" # For "Your bilingual companion..."
    label_color = "#ffffff"  # For "How are you feeling..."
    card_bg = "rgba(255, 255, 255, 0.05)"
    border_color = "#4caf50"
    btn_bg = "#333333"
    btn_hover = "#444444"
else:
    bg_color = "#F7F9F7"
    title_color = "#1b5e20"
    text_color = "#212121"
    subtext_color = "#444444" 
    label_color = "#1b5e20"
    card_bg = "#ffffff"
    border_color = "#2e7d32"
    btn_bg = "#eeeeee"
    btn_hover = "#dddddd"

st.markdown(f"""
<link href="https://cdn.jsdelivr.net/npm/jameel-noori@1.1.2/jameel-noori.min.css" rel="stylesheet">
<style>
    /* Force background and global text visibility */
    .stApp {{
        background-color: {bg_color};
        transition: 0.3s;
    }}
    
    /* Fix for "Your bilingual companion..." text */
    .subtext {{
        color: {subtext_color} !important;
        font-style: italic;
        margin-bottom: 20px;
        display: block;
    }}

    /* Fix for "How are you feeling today?" label */
    label p {{
        color: {label_color} !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
    }}

    h1 {{
        color: {title_color} !important;
    }}

    .english-font {{
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 19px;
        color: {text_color};
        line-height: 1.6;
    }}

    .urdu-font {{
        font-family: 'Jameel Noori', 'Jameel Noori Nastaleeq', serif;
        direction: rtl;
        text-align: right;
        font-size: 28px;
        line-height: 1.8;
        color: #2e7d32;
    }}

    .source-box {{
        background-color: {card_bg};
        border: 2px solid {border_color};
        border-radius: 15px;
        padding: 25px;
        margin: 25px 0;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}

    /* Fix for Toggle Button visibility and hover */
    button[kind="secondary"] {{
        background-color: {btn_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    button[kind="secondary"]:hover {{
        background-color: {btn_hover} !important;
        color: {title_color} !important;
        border: 1px solid {title_color} !important;
    }}

    .source-label {{
        font-size: 14px;
        color: {border_color};
        font-weight: bold;
        text-transform: uppercase;
        display: block;
        margin-bottom: 12px;
    }}
</style>
""", unsafe_allow_html=True)

# --- 4. INITIALIZE MODELS ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("sukoon_index1", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    return vector_db, llm

vector_db, llm = load_resources()

# --- 5. THE UI ---
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown(f"<h1>Sukoon AI (سکون)</h1>", unsafe_allow_html=True)
with col2:
    mode_label = "☀️ Light" if st.session_state.dark_mode else "🌙 Night"
    st.button(mode_label, on_click=toggle_mode)

# Using a custom class to ensure visibility in both modes
st.markdown(f'<span class="subtext">🌿 Your bilingual companion for spiritual peace.</span>', unsafe_allow_html=True)

user_input = st.text_input("How are you feeling today? / آپ کیسا محسوس کر رہے ہیں؟")

if user_input:
    if any(word in user_input.lower() for word in ["suicide", "hurt", "die", "khudkushi", "marna"]):
        st.error("Please reach out to Umang Helpline (Lahore): 0311-7786264 immediately.")
    else:
        with st.spinner("Finding sukoon for you..."):
            docs = vector_db.similarity_search(user_input, k=1)
            context = docs[0].page_content

            prompt = f"""
            You are Sukoon AI.
            Context: {context}
            User Input: {user_input}
            Format:
            ENG_PART: [English message]
            VERSE_PART: [Hadith or Ayah]
            URDU_PART: [Urdu message]
            """
            
            response = llm.invoke(prompt).content

            try:
                eng_text = response.split("ENG_PART:")[1].split("VERSE_PART:")[0].strip()
                verse_text = response.split("VERSE_PART:")[1].split("URDU_PART:")[0].strip()
                urdu_text = response.split("URDU_PART:")[1].strip()

                st.markdown(f'<div class="english-font">{eng_text}</div>', unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="source-box">
                        <span class="source-label">Divine Guidance / وحی کی روشنی</span>
                        <div class="urdu-font" style="text-align: center;">{verse_text}</div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f'<div class="urdu-font">{urdu_text}</div>', unsafe_allow_html=True)

            except:
                st.write(response)

st.caption("Sukoon AI provides spiritual support. For clinical emergencies, consult a professional.")
