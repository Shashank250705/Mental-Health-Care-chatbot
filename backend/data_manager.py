import pandas as pd
from datasets import load_dataset

def load_mental_health_data():
    """
    Load the mental health dataset from Hugging Face
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    # Load dataset from Hugging Face
    dataset = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset['train'])
    
    return df