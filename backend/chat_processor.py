import streamlit as st
from utils.preprocessing import preprocess_text
from utils.response_generator import generate_response

def process_user_input(prompt, df, model_info):
    """
    Process user input and generate a response
    
    Args:
        prompt (str): The user's input message
        df (pandas.DataFrame): The mental health dataset
        model_info (dict): Information about the model to use
        
    Returns:
        None: Updates the session state and UI directly
    """
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Preprocess the user input
    processed_input = preprocess_text(prompt)
    
    if model_info:
        # Generate response using the Groq API
        response = generate_response(processed_input, df, model_info, None)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        return response
    else:
        st.error("Could not access the Groq API. Please check your API key.")
        return None