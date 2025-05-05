import os
import requests
import streamlit as st

def transcribe_audio(audio_bytes, api_key):
    """
    Transcribe audio using Whisper large-v3 model via Groq API
    
    Args:
        audio_bytes (bytes): The audio data to transcribe
        api_key (str): The Groq API key
        
    Returns:
        str: The transcribed text
    """
    # Create a temporary file to store the audio
    temp_file = "temp_audio.wav"
    with open(temp_file, "wb") as f:
        f.write(audio_bytes)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "multipart/form-data"
    }
    
    try:
        # Open the file for the multipart/form-data request
        with open(temp_file, "rb") as f:
            files = {
                "file": (temp_file, f, "audio/wav")
            }
            
            data = {
                "model": "whisper-large-v3",
                "language": "en"
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data
            )
        
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        if response.status_code == 200:
            result = response.json()
            return result.get("text", "")
        else:
            st.error(f"Error transcribing audio: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        st.error(f"Exception during transcription: {str(e)}")
        # Clean up the temporary file in case of exception
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return ""