import streamlit as st
import os

def setup_groq_api():
    """
    Set up access to the Groq API
    
    Returns:
        dict: API configuration with key and model name, or None if setup fails
    """
    # Try to get API key from Streamlit secrets first
    try:
        api_key = st.secrets["groq_api_key"]
    except:
        # If not found in secrets, try environment variables
        api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("No API key found for Groq. Please check your secrets or environment variables.")
        return None
    
    # Return the API key and model name
    return {"api_key": api_key, "model": "llama3-70b-8192"}