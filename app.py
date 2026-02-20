import streamlit as st
import streamlit_authenticator as stauth
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Sukoon AI", page_icon="🌿", layout="centered")

# --- 2. AUTHENTICATION LOGIC ---
# Initializing credentials in session state for demo purposes.
# In a professional app, move these to secrets or a database.
if 'credentials' not in st.session_state:
    st.session_state.credentials = {
        'usernames': {
            'user123': {
                'email': 'user@example.com',
                'name': 'Sukoon User',
                'password': '123'  # Note: Use hashed passwords in production
            }
        }
    }

authenticator = stauth.Authenticate(
    st.session_state.credentials,
    'sukoon_cookie',
    'sukoon_key',
    30
)

# --- 3. LOGIN & SIGNUP UI ---
if not st.session_state.get("authentication_status"):
    st.markdown("<h1 style='text-align: center; color: #2e7d32;'>Sukoon AI | سکون</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with tab1:
        # Latest version uses location='main'
        authenticator.login(location='main')
    
    with tab2:
        try:
            # UPDATED: Changed 'pre_authorization' to 'pre_authorized'
            # Also capturing the result to update the credentials
            if authenticator.register_user(location='main', pre_authorized=False):
                st.success('User registered successfully! Please switch to the Login tab.')
                # This ensures the new user is added to the session state immediately
                st.session_state.credentials = authenticator.credentials
        except Exception as e:
            st.error(f"Registration error: {e}")

# --- 4. PROTECTED APP CONTENT (Only visible if logged in) ---
if st.session_state.get("authentication_status"):
    
    # --- THEME STATE ---
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

    def toggle_mode():
        st.session_state.dark_mode = not st.session_state.dark_mode

    # --- DYNAMIC STYLING ---
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
    <link href="https://cdn.jsdelivr.net/npm/jameel-noori@1.1.2/jameel-noori.min.css" rel="stylesheet">
    <style>
        .stApp {{ background-color: {bg_color}; transition: 0.3s; }}
        .subtext {{ color: {subtext_color} !important; font-style: italic; display: block; margin-bottom: 20px; }}
        label p {{ color: {label_color} !important; font-weight: bold !important; font-size: 1.1rem !important; }}
        h1 {{ color: {title_color} !important; font-family: 'Source Sans Pro', sans-serif; }}
        .english-font {{ font-family: 'Source Sans Pro', sans-serif; font-size: 19px; color: {text_color}; line-height: 1.6; }}
        .urdu-font {{ font-family: 'Jameel Noori', 'Jameel Noori Nastaleeq', serif; direction: rtl; text-align: right; font-size: 28px; line-height: 1.8; color: #2e7d32; }}
        .source-box {{ background-color: {card_bg}; border: 2px solid {border_color}; border-radius: 15px; padding: 25px; margin: 25px 0; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .source-label {{ font-size: 14px; color: {border_color}; font-weight: bold; text-transform: uppercase; display: block; margin-bottom: 12px; }}
        button[kind="secondary"] {{ background-color: {btn_bg} !important; color: {text_color} !important; border: 1px solid {border_color} !important; }}
    </style>
    """, unsafe_allow_html=True)

    # --- TOP BAR ---
    col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
    with col1:
        st.markdown(f"<h1>Sukoon AI</h1>", unsafe_allow_html=True)
    with col2:
        mode_label = "☀️ Light" if st.session_state.dark_mode else "🌙 Night"
        st.button(mode_label, on_click=toggle_mode)
    with col3:
        authenticator.logout(button_name='Logout', location='main')

    st.markdown(f'<span class="subtext">🌿 Welcome, {st.session_state["name"]}. Your spiritual sanctuary.</span>', unsafe_allow_html=True)

    # --- RESOURCE LOADING ---
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

    @st.cache_resource
    def load_resources():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.load_local("sukoon_index1", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
        return vector_db, llm

    vector_db, llm = load_resources()

    # --- MAIN INTERFACE ---
    user_input = st.text_input("How are you feeling today? / آپ کیسا محسوس کر رہے ہیں؟")

    if user_input:
        if any(word in user_input.lower() for word in ["suicide", "hurt", "die", "khudkushi", "marna"]):
            st.error("Please reach out to Umang Helpline (Lahore): 0311-7786264 immediately.")
        else:
            with st.spinner("Finding sukoon for you..."):
                docs = vector_db.similarity_search(user_input, k=1)
                context = docs[0].page_content
                
                prompt = f"""You are Sukoon AI. Use Context: {context}. 
                Format your response exactly:
                ENG_PART: [English comfort message]
                VERSE_PART: [Hadith or Ayah]
                URDU_PART: [Urdu explanation]"""
                
                try:
                    response = llm.invoke(f"{prompt} User Input: {user_input}").content
                    
                    # Splitting logic
                    eng = response.split("ENG_PART:")[1].split("VERSE_PART:")[0].strip()
                    verse = response.split("VERSE_PART:")[1].split("URDU_PART:")[0].strip()
                    urdu = response.split("URDU_PART:")[1].strip()

                    # Display
                    st.markdown(f'<div class="english-font">{eng}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="source-box"><span class="source-label">Divine Guidance</span><div class="urdu-font" style="text-align:center;">{verse}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="urdu-font">{urdu}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.info("Direct Guidance:")
                    st.write(response)

    st.caption("Sukoon AI | Managed Securely")

# --- 5. ERROR HANDLING ---
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
