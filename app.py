import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Sukoon AI", page_icon="🌿", layout="centered")

# Load Jameel Noori Nastaleeq for a beautiful Urdu experience
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
    /* New Beautiful Container for Hadith/Ayah */
    .source-box {
        background-color: rgba(46, 125, 50, 0.05);
        border: 1px solid #2e7d32;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .source-label {
        font-size: 12px;
        color: #2e7d32;
        font-weight: bold;
        text-transform: uppercase;
        display: block;
        margin-bottom: 10px;
    }
    hr { margin: 20px 0; border: 0; border-top: 1px solid #eee; }
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

            # C. UPDATED PROMPT: Forces AI to use markers for splitting
            prompt = f"""
            You are Sukoon AI, a gentle spiritual mentor.
            Context: {context}
            User Input: {user_input}

            Format your response exactly as follows:
            ENG_PART: [Your comforting English message]
            VERSE_PART: [The specific Hadith or Ayah in Arabic/Urdu from the context]
            URDU_PART: [Your comforting Urdu translation/message]
            """
            
            response = llm.invoke(prompt).content

            try:
                # D. DISPLAY LOGIC: Splitting by markers
                eng_text = response.split("ENG_PART:")[1].split("VERSE_PART:")[0].strip()
                verse_text = response.split("VERSE_PART:")[1].split("URDU_PART:")[0].strip()
                urdu_text = response.split("URDU_PART:")[1].strip()

                # 1. Display English Guidance
                st.markdown(f'<div class="english-font">{eng_text}</div>', unsafe_allow_html=True)

                # 2. Display Beautiful Verse Container
                st.markdown(f"""
                    <div class="source-box">
                        <span class="source-label">Divine Guidance / وحی کی روشنی</span>
                        <div class="urdu-font" style="text-align: center; color: #1b5e20;">{verse_text}</div>
                    </div>
                """, unsafe_allow_html=True)

                # 3. Display Urdu Guidance
                st.markdown(f'<div class="urdu-font">{urdu_text}</div>', unsafe_allow_html=True)

            except:
                # Safe fallback if AI messes up the format
                st.info("Guidance Found:")
                st.write(response)

# --- 4. FOOTER ---
st.caption("Sukoon AI provides spiritual support. For clinical emergencies, consult a professional.")
