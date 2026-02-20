import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Sukoon AI", page_icon="🌿", layout="centered")

# Load Jameel Noori Nastaleeq and styling
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/jameel-noori@1.1.2/jameel-noori.min.css" rel="stylesheet">
    <style>
    .urdu-font {
        font-family: 'Jameel Noori', 'Jameel Noori Nastaleeq', serif;
        direction: rtl;
        text-align: right;
        font-size: 24px;
        line-height: 1.6;
        color: #2e7d32;
    }
    .english-font {
        font-family: 'Source Sans Pro', sans-serif;
        direction: ltr;
        text-align: left;
        font-size: 18px;
        color: #333;
    }
    /* Style for the Label above the Hadith box */
    .source-label {
        font-size: 13px;
        color: #2e7d32;
        font-weight: bold;
        text-transform: uppercase;
        margin-top: 20px;
        margin-bottom: 5px;
        display: block;
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
st.write("🌿 *Your bilingual companion for spiritual peace.*")

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

            Format your response exactly as follows:
            ENG_PART: [English message]
            VERSE_PART: [Hadith or Ayah text only]
            URDU_PART: [Urdu message]
            """
            
            response = llm.invoke(prompt).content

            try:
                # Splitting the response
                eng_text = response.split("ENG_PART:")[1].split("VERSE_PART:")[0].strip()
                verse_text = response.split("VERSE_PART:")[1].split("URDU_PART:")[0].strip()
                urdu_text = response.split("URDU_PART:")[1].strip()

                # 1. Display English
                st.markdown(f'<div class="english-font">{eng_text}</div>', unsafe_allow_html=True)

                # 2. Display Native Streamlit Copy-able Box
                st.markdown('<span class="source-label">Divine Guidance / وحی کی روشنی</span>', unsafe_allow_html=True)
                # This 'st.code' block has a built-in copy button that works everywhere
                st.code(verse_text, language=None)

                # 3. Display Urdu
                st.markdown(f'<div class="urdu-font">{urdu_text}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.write(response)

# --- 4. FOOTER ---
st.caption("Sukoon AI provides spiritual support. For clinical emergencies, consult a professional.")
