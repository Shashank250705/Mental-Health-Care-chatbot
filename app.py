import os
# Set these environment variables before importing any other modules
os.environ["STREAMLIT_WATCHED_MODULES"] = "none"  # Disable all module watching
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"  # Disable usage stats
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false"  # Disable static serving

# Import streamlit after setting environment variables
import streamlit as st

# Import frontend components
from frontend.ui import (
    set_page_config,
    load_css,
    render_header,
    render_disclaimer,
    render_faq,
    render_chat_interface,
    display_user_message,
    display_assistant_message
)

# Import backend components
from backend.data_manager import load_mental_health_data
from backend.api_manager import setup_groq_api
from backend.audio_processor import transcribe_audio
from backend.chat_processor import process_user_input

# Set page configuration - KEEP THIS AS THE FIRST STREAMLIT COMMAND
set_page_config()

# Apply custom styling
load_css()

def main():
    # Render header
    render_header()
    
    # Render disclaimer
    render_disclaimer()
    
    # Render FAQ section
    render_faq()
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Render chat interface
    audio_file, text_input = render_chat_interface()
    
    # Load data and model info
    df = load_mental_health_data()
    model_info = setup_groq_api()
    
    # Process audio if uploaded
    if audio_file is not None:
        with st.spinner("Transcribing your message..."):
            if model_info:
                # Read the audio file
                audio_bytes = audio_file.read()
                
                # Transcribe audio
                transcription = transcribe_audio(audio_bytes, model_info["api_key"])
                if transcription:
                    st.success(f"Transcribed: {transcription}")
                    
                    # Display user message
                    display_user_message(transcription)
                    
                    # Process the transcribed text
                    with st.spinner("Thinking..."):
                        response = process_user_input(transcription, df, model_info)
                        if response:
                            display_assistant_message(response)
            else:
                st.error("Could not access the Groq API. Please check your API key.")
    
    # Process text input
    if text_input:
        # Display user message
        display_user_message(text_input)
        
        # Process user input
        with st.spinner("Thinking..."):
            response = process_user_input(text_input, df, model_info)
            if response:
                display_assistant_message(response)

if __name__ == "__main__":
    main()