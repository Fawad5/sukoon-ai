import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. THE BEAUTY LAYER (CSS) ---
st.set_page_config(page_title="Sukoon AI", page_icon="🌿")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Urdu:wght@400;700&display=swap');
    
    .main-card {
        border-radius: 15px;
        padding: 20px;
        background-color: #f0f4f2;
        border-left: 5px solid #2e7d32;
        margin-bottom: 20px;
    }
    
    .english-text {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 18px;
        color: #2c3e50;
        line-height: 1.6;
        text-align: left;
        direction: ltr;
    }
    
    .urdu-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        border-right: 5px solid #1b5e20;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    
    .urdu-text {
        font-family: 'Noto Sans Urdu', serif;
        font-size: 22px;
        color: #1b5e20;
        line-height: 2.0;
        text-align: right;
        direction: rtl;
    }
    
    .hadith-label {
        font-weight: bold;
        color: #1b5e20;
        font-size: 14px;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC (Assuming resources are loaded as before) ---
# [Insert your load_resources() function here]

st.title("Sukoon AI | سکون")
user_input = st.text_input("Share your heart... | اپنے دل کی بات بتائیں...")

if user_input:
    with st.spinner("Reflecting..."):
        # Search and Generate
        docs = vector_db.similarity_search(user_input, k=1)
        context = docs[0].page_content
        
        # Updated Prompt to force a clear split
        prompt = f"""
        User Input: {user_input}
        Spiritual Context: {context}
        
        Role: Gentle Mentor. 
        Task: Provide comfort.
        
        Format:
        English: [Comforting message]
        Hadith_Urdu: [The Urdu text of the Hadith/Ayah]
        Explanation_Urdu: [A gentle Urdu explanation]
        """
        
        response = llm.invoke(prompt).content

        # --- 3. BEAUTIFUL DISPLAY LOGIC ---
        # Splitting the response manually to avoid messiness
        try:
            # We assume the AI follows the format. We split by keywords.
            parts = response.split("English:")[1].split("Hadith_Urdu:")
            english_msg = parts[0].strip()
            urdu_parts = parts[1].split("Explanation_Urdu:")
            hadith_urdu = urdu_parts[0].strip()
            explanation_urdu = urdu_parts[1].strip()

            # Display English Card
            st.markdown(f"""
                <div class="main-card">
                    <div class="english-text">{english_msg}</div>
                </div>
            """, unsafe_allow_html=True)

            # Display Urdu Card
            st.markdown(f"""
                <div class="urdu-card">
                    <div class="hadith-label">قرآنی رہنمائی / حدیث مبارکہ:</div>
                    <div class="urdu-text">{hadith_urdu}</div>
                    <hr>
                    <div class="urdu-text" style="color: #444; font-size: 20px;">{explanation_urdu}</div>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            # Simple fallback if splitting fails
            st.write(response)
