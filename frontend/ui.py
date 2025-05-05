import streamlit as st
import os

def set_page_config():
    """Set page configuration"""
    # Note: This should only be called once in the main app
    st.set_page_config(
        page_title="Mental Health Care Chatbot",
        page_icon="🧠",
        layout="wide"
    )

def load_css():
    """Load custom CSS"""
    css_path = os.path.join(os.path.dirname(__file__), "static", "css", "style.css")
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_header():
    """Render the page header"""
    st.title("Mental Health Care Chatbot")

def render_disclaimer():
    """Render the disclaimer section"""
    with st.container():
        st.markdown("""
        <div class="disclaimer">
        <h3>Welcome to your Mental Health Support Assistant</h3>
        <p>This chatbot is designed to provide mental health support and resources. 
        Please note that this is not a substitute for professional mental health care.</p>
        <p>If you're experiencing a crisis, please contact emergency services or a mental health professional.</p>
        </div>
        """, unsafe_allow_html=True)

def render_faq():
    """Render the FAQ section"""
    with st.expander("📋 Frequently Asked Questions", expanded=False):
        st.markdown("""
        <div class="faq-section">
            <div class="faq-item">
                <h4>What can this chatbot help me with?</h4>
                <p>This chatbot can provide information about mental health topics, offer coping strategies for common issues like anxiety and stress, and suggest resources for further support.</p>
            </div>
            <div class="faq-item">
                <h4>Is my conversation private?</h4>
                <p>Yes, your conversations are not stored permanently and are only used to provide you with appropriate responses during your current session.</p>
            </div>
            <div class="faq-item">
                <h4>What should I do in a mental health emergency?</h4>
                <p>If you're experiencing a mental health emergency or having thoughts of self-harm, please contact emergency services (911) or a Crisis helpline immediately. In the US, you can call the National Suicide Prevention Lifeline at 988.</p>
            </div>
            <div class="faq-item">
                <h4>How accurate is the information provided?</h4>
                <p>The chatbot uses reliable mental health information, but it's not a substitute for professional advice. Always consult with a qualified mental health professional for personalized guidance.</p>
            </div>
            <div class="faq-item">
                <h4>Can I ask questions about specific mental health conditions?</h4>
                <p>Yes, you can ask about various mental health conditions, symptoms, and general coping strategies. The chatbot will provide informational support based on available resources.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_chat_interface():
    """Render the chat interface with audio upload and text input"""
    # Create columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Add audio input option using file uploader
        st.markdown("<h3 style='text-align: center;'>Voice Input</h3>", unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload audio message", type=['wav', 'mp3', 'm4a'])
    
    with col1:
        # Display chat history with custom styling
        if "messages" in st.session_state:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(f"<div class='{message['role']}-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Return the audio file and text input
    text_input = st.chat_input("How are you feeling today?")
    
    return audio_file, text_input

def display_user_message(message):
    """Display a user message with styling"""
    with st.chat_message("user"):
        st.markdown(f"<div class='user-message'>{message}</div>", unsafe_allow_html=True)

def display_assistant_message(message):
    """Display an assistant message with styling"""
    with st.chat_message("assistant"):
        st.markdown(f"<div class='assistant-message'>{message}</div>", unsafe_allow_html=True)