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
    hr { margin: 20px 0; border: 0; border-top: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALIZE MODELS ---
# Securely get API key from Streamlit Cloud Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def load_resources():
    # Free local embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Load your local "Memory"
    vector_db = FAISS.load_local("sukoon_index1", embeddings, allow_dangerous_deserialization=True)
    # Fast free AI brain
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    return vector_db, llm

vector_db, llm = load_resources()

# --- 3. THE UI ---
st.title("Sukoon AI (سکون)")
st.write("🌿 *Your bilingual companion for spiritual peace.*")

user_input = st.text_input("How are you feeling today? / آپ کیسا محسوس کر رہے ہیں؟")

if user_input:
    # A. Safety Check
    if any(word in user_input.lower() for word in ["suicide", "hurt", "die", "khudkushi", "marna"]):
        st.error("Please reach out to Umang Helpline (Lahore): 0311-7786264 immediately.")
    else:
        with st.spinner("Finding sukoon for you..."):
            # B. Search local Islamic Library
            docs = vector_db.similarity_search(user_input, k=1)
            context = docs[0].page_content

            # C. Bilingual System Instruction
            # We tell the AI exactly how to split the response
            prompt = f"""
            You are Sukoon AI, a gentle spiritual mentor for youth. 
            For every response:
            1. Provide a comforting message in English.
            2. Provide the same message translated into clear, soft Urdu.
            3. Always include the specific Hadith or Ayah in both Arabic (if available) and Urdu.
            4. Format the output clearly so English is at the top and Urdu is at the bottom.
            """
            
            response = llm.invoke(prompt).content

            # D. Display Results
            try:
                # Split the AI's response into the two languages
                eng_part, urdu_part = response.split("SEPARATOR")
                
                # Display English
                st.markdown(f'<div class="english-font">{eng_part.replace("ENGLISH:", "").strip()}</div>', unsafe_allow_html=True)
                
                st.markdown("<hr>", unsafe_allow_html=True)
                
                # Display Urdu with proper font and RTL direction
                st.markdown(f'<div class="urdu-font">{urdu_part.replace("URDU:", "").strip()}</div>', unsafe_allow_html=True)
            
            except:
                # Fallback if AI forgets the separator
                st.write(response)

# --- 4. FOOTER ---
st.caption("Sukoon AI provides spiritual support. For clinical emergencies, consult a professional.")
