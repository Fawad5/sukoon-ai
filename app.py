import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIG ---
st.set_page_config(page_title="Sukoon AI", page_icon="🌿")

# CUSTOM CSS: This makes Urdu text align to the right (RTL support)
st.markdown("""
    <style>
    .rtl-text { direction: RTL; text-align: right; font-family: 'Jameel Noori Nastaleeq'; font-size: 20px; }
    </style>
    """, unsafe_allow_html=True)

# Access the key securely from the cloud settings
groq_key = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    api_key=groq_key, 
    model_name="llama-3.3-70b-versatile"
)

# 2. Load your Local Library (Add a spinner here)
with st.spinner("Waking up Sukoon AI's memory..."):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("sukoon_index1", embeddings, allow_dangerous_deserialization=True)
st.success("Ready to provide guidance!")

# --- THE UI ---
st.title("Sukoon AI (سکون)")
st.write("Find spiritual peace and guidance.")

user_input = st.text_input("How are you feeling?")

if user_input:
    # A. Check for Emergency (Safety first!)
    if any(word in user_input.lower() for word in ["suicide", "hurt", "die", "khudkushi"]):
        st.error("Please reach out to Umang Helpline (Lahore): 0311-7786264 immediately.")
    else:
        # B. Search your local library
        docs = vector_db.similarity_search(user_input, k=1)
        best_match = docs[0].page_content
        
        # C. Generate compassionate response
        prompt = f"User: {user_input}\nRetrieved Guidance: {best_match}\nAct as a gentle mentor. Respond with empathy."
        response = llm.invoke(prompt)
        
        # D. Display response with Urdu support
        st.markdown(f'<div class="rtl-text">{response.content}</div>', unsafe_allow_html=True)

