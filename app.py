import streamlit as st
import streamlit_authenticator as stauth
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Sukoon AI", page_icon="🌿", layout="centered")

# --- 2. SESSION STATE INITIALIZATION ---
# We must ensure these exist before the authenticator is created
if 'credentials' not in st.session_state:
    st.session_state.credentials = {
        'usernames': {
            'user123': {'email': 'u@s.com', 'name': 'User', 'password': '123'}
        }
    }

# --- 3. AUTHENTICATOR SETUP ---
# Newer versions require the 'cookie' and 'pre_authorized' to be explicitly defined
authenticator = stauth.Authenticate(
    credentials=st.session_state.credentials,
    cookie_name='sukoon_app_cookie',
    key='signature_key_123',
    cookie_expiry_days=30,
    pre_authorized=[] # Mandatory list
)

# --- 4. UI LOGIC ---
# Using a simpler logic check to avoid "Blank Screen" loops
if not st.session_state.get("authentication_status"):
    st.markdown("<h1 style='text-align: center;'>Sukoon AI</h1>", unsafe_allow_html=True)
    
    tab_choice = st.radio("Select Action", ["Login", "Sign Up"], horizontal=True)
    
    if tab_choice == "Login":
        # Simplified login call
        authenticator.login(location='main')
        
        if st.session_state.get("authentication_status") is False:
            st.error("Incorrect username/password")
        elif st.session_state.get("authentication_status") is None:
            st.info("Please enter your details")
            
    else:
        try:
            # Registration logic
            if authenticator.register_user(location='main'):
                # Version-safe credential syncing
                if hasattr(authenticator, 'authenticator_dict'):
                    st.session_state.credentials = authenticator.authenticator_dict
                else:
                    st.session_state.credentials = authenticator.credentials
                
                st.success("Success! Please switch to Login.")
                time.sleep(1)
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# --- 5. MAIN APP CONTENT ---
if st.session_state.get("authentication_status"):
    st.title(f"Welcome to Sukoon, {st.session_state['name']}!")
    
    # Logout button
    authenticator.logout(location='main')
    
    st.divider()
    st.write("Your protected AI content goes here.")
