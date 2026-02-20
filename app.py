import streamlit as st
import streamlit_authenticator as stauth
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

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
authenticator = stauth.Authenticate(
    st.session_state.credentials,
    'sukoon_cookie',
    'sukoon_key',
    30,
    pre_authorized=[]  # ADD THIS LINE HERE
)

# --- 4. LOGIN & SIGNUP UI ---
# We check the status at the very beginning
auth_status = st.session_state.get("authentication_status")

if not auth_status:
    st.markdown("<h1 style='text-align: center; color: #2e7d32;'>Sukoon AI | سکون</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with tab1:
        # We handle login here. 'fields' argument helps prevent some default messages
        authenticator.login(location='main')
        
        # Only show messages if the user HAS attempted to log in
        if st.session_state.get("authentication_status") is False:
            st.error('Username/password is incorrect')
        elif st.session_state.get("authentication_status") is None:
            st.info('Please enter your credentials to find sukoon.')
    
    with tab2:
        try:
            # REMOVE the pre_authorized=[] from here
            reg_result = authenticator.register_user(location='main')
            
            if reg_result:
                st.success('User registered successfully! Please go to the Login tab.')
                st.session_state.credentials = authenticator.authenticator_dict
        except Exception as e:
            # This handles the "User not pre-authorized" or other errors
            st.error(f"Registration error: {e}")

# --- 5. MAIN PROTECTED APP CONTENT ---
if st.session_state.get("authentication_status"):
    
    # Theme Management
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

    def toggle_mode():
        st.session_state.dark_mode = not st.session_state.dark_mode

    # Dynamic styling (Same as before)
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
        h1 {{ color: {title_color} !important; }}
        .subtext {{ color: {subtext_color} !important; font-style: italic; }}
        .english-font {{ font-family: 'Source Sans Pro', sans-serif; font-size: 19px; color: {text_color}; }}
        .urdu-font {{ font-family: 'Jameel Noori Nastaleeq', serif; direction: rtl; text-align: right; font-size: 28px; color: #2e7d32; }}
        .source-box {{ background-color: {card_bg}; border: 2px solid {border_color}; border-radius: 15px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        label p {{ color: {label_color} !important; font-weight: bold !important; }}
        button[kind="secondary"] {{ background-color: {btn_bg} !important; color: {text_color} !important; border: 1px solid {border_color} !important; }}
    </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
    with col1:
        st.markdown(f"<h1>Sukoon AI</h1>", unsafe_allow_html=True)
    with col2:
        st.button("☀️ Light" if st.session_state.dark_mode else "🌙 Night", on_click=toggle_mode)
    with col3:
        authenticator.logout(location='main')

    st.markdown(f'<span class="subtext">🌿 Welcome, {st.session_state.get("name")}. Your spiritual sanctuary.</span>', unsafe_allow_html=True)

    # Resource Loading
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

    @st.cache_resource
    def load_resources():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Ensure your index folder 'sukoon_index1' is in the same directory
        vector_db = FAISS.load_local("sukoon_index1", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
        return vector_db, llm

    vector_db, llm = load_resources()

    # Chat UI
    user_input = st.text_input("How are you feeling today? / آپ کیسا محسوس کر رہے ہیں؟")

    if user_input:
        if any(word in user_input.lower() for word in ["suicide", "hurt", "die", "khudkushi", "marna"]):
            st.error("Please reach out to Umang Helpline: 0311-7786264.")
        else:
            with st.spinner("Finding sukoon..."):
                docs = vector_db.similarity_search(user_input, k=1)
                context = docs[0].page_content
                prompt = f"Context: {context}. Format: ENG_PART: [Text] VERSE_PART: [Text] URDU_PART: [Text]"
                try:
                    response = llm.invoke(f"{prompt} User Input: {user_input}").content
                    eng = response.split("ENG_PART:")[1].split("VERSE_PART:")[0].strip()
                    verse = response.split("VERSE_PART:")[1].split("URDU_PART:")[0].strip()
                    urdu = response.split("URDU_PART:")[1].strip()
                    st.markdown(f'<div class="english-font">{eng}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="source-box"><div class="urdu-font" style="text-align:center;">{verse}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="urdu-font">{urdu}</div>', unsafe_allow_html=True)
                except:
                    st.write(response)

    st.caption("Sukoon AI | Powered by Faith & AI")
