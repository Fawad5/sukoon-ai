import streamlit as st
import streamlit_authenticator as stauth
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Sukoon AI", page_icon="🌿", layout="centered")

# --- 2. INITIALIZE SESSION STATE ---
if 'credentials' not in st.session_state:
    st.session_state.credentials = {
        'usernames': {
            'user123': {
                'email': 'user@example.com',
                'name': 'Sukoon User',
                'password': '123'
            }
        }
    }

# --- 3. AUTHENTICATION SETUP ---
# Defining pre_authorized=[] here is the KEY to making the Sign Up tab show up
authenticator = stauth.Authenticate(
    st.session_state.credentials,
    'sukoon_cookie',
    'sukoon_key',
    30,
    pre_authorized=[]
)

# --- 4. LOGIN & SIGNUP UI ---
auth_status = st.session_state.get("authentication_status")

if not auth_status:
    st.markdown("<h1 style='text-align: center; color: #2e7d32;'>Sukoon AI | سکون</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with tab1:
        authenticator.login(location='main')
        if st.session_state.get("authentication_status") is False:
            st.error('Username/password is incorrect')
        elif st.session_state.get("authentication_status") is None:
            st.info('Please enter your credentials to find sukoon.')
            
    with tab2:
        try:
            # This function renders the fields. If pre_authorized=[] was 
            # defined above, this will now appear correctly.
            reg_result = authenticator.register_user(location='main')
            
            if reg_result:
                # Universal attribute finder to sync the new user
                if hasattr(authenticator, 'authenticator_dict'):
                    st.session_state.credentials = authenticator.authenticator_dict
                else:
                    st.session_state.credentials = authenticator.credentials
                
                st.success('User registered successfully!')
                st.info('Refreshing for Login...')
                time.sleep(1.5)
                st.rerun()
        except Exception as e:
            # Hiding background noise while the form is empty
            if "NoneType" not in str(e) and "must not be None" not in str(e):
                st.error(f"Registration error: {e}")

# --- 5. PROTECTED APP CONTENT ---
if st.session_state.get("authentication_status"):
    
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

    def toggle_mode():
        st.session_state.dark_mode = not st.session_state.dark_mode

    # Theme logic
    if st.session_state.dark_mode:
        bg_color, title_color, text_color = "#121212", "#4caf50", "#e0e0e0"
        subtext_color, label_color = "#aaaaaa", "#ffffff"
        card_bg, border_color = "rgba(255, 255, 255, 0.05)", "#4caf50"
        btn_bg, btn_hover = "#333333", "#444444"
    else:
        bg_color, title_color, text_color = "#F7F9F7", "#1b5e20", "#212121"
        subtext_color, label_color = "#444444", "#1b5e20"
        card_bg, border_color = "#ffffff", "#2e7d32"
        btn_bg, btn_hover = "#eeeeee", "#dddddd"

    st.markdown(f"""
    <style>
        .stApp {{ background-color: {bg_color}; transition: 0.3s; }}
        h1 {{ color: {title_color} !important; }}
        .english-font {{ font-family: 'Source Sans Pro', sans-serif; font-size: 19px; color: {text_color}; }}
        .urdu-font {{ font-family: 'serif'; direction: rtl; text-align: right; font-size: 28px; color: #2e7d32; }}
        .source-box {{ background-color: {card_bg}; border: 2px solid {border_color}; border-radius: 15px; padding: 20px; }}
    </style>
    """, unsafe_allow_html=True)

    # Header with Logout
    col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
    with col1:
        st.markdown(f"<h1>Sukoon AI</h1>", unsafe_allow_html=True)
    with col2:
        st.button("☀️ Light" if st.session_state.dark_mode else "🌙 Night", on_click=toggle_mode)
    with col3:
        authenticator.logout(location='main')

    # AI Logic
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

    @st.cache_resource
    def load_resources():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.load_local("sukoon_index1", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
        return vector_db, llm

    vector_db, llm = load_resources()

    user_input = st.text_input("How are you feeling today? / آپ کیسا محسوس کر رہے ہیں؟")

    if user_input:
        with st.spinner("Finding sukoon..."):
            docs = vector_db.similarity_search(user_input, k=1)
            context = docs[0].page_content
            prompt = f"Context: {context}. Format response with ENG_PART:, VERSE_PART:, and URDU_PART:."
            response = llm.invoke(f"{prompt} User: {user_input}").content
            st.write(response)
