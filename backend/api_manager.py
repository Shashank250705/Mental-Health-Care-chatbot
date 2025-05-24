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
        api_key = st.secrets.get("groq_api_key", "")
        st.sidebar.info("Attempting to access API key from Streamlit secrets")
    except Exception as e:
        st.sidebar.error(f"Error accessing Streamlit secrets: {str(e)}")
        api_key = ""
    
    # If not found in secrets, try environment variables
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            st.sidebar.success("Found API key in environment variables")
        
    if not api_key:
        st.error("""
        No API key found for Groq. Please check:
        
        1. For Streamlit Cloud: Verify the API key is correctly set in the Secrets section
        2. For local development: Check your .streamlit/secrets.toml file
        3. For other platforms: Ensure the GROQ_API_KEY environment variable is set
        
        Note: Changes to secrets may take a minute to propagate.
        """)
        return None
    
    # Return the API key and model name
    return {"api_key": api_key, "model": "llama3-70b-8192"}