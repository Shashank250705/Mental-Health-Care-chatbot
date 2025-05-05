import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

def load_and_preprocess_dataset(dataset_name="heliosbrahma/mental_health_chatbot_dataset"):
    """
    Load and preprocess the mental health dataset from Hugging Face
    """
    print(f"Loading dataset from Hugging Face: {dataset_name}")
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_name)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset['train'])
        
        # Basic preprocessing
        # Fill missing values
        df = df.fillna('')
        
        # Remove duplicates if any
        df = df.drop_duplicates()
        
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def get_dataset_info(df):
    """
    Get basic information about the dataset
    """
    info = {
        "num_rows": len(df),
        "columns": list(df.columns),
        "sample_data": df.head(3).to_dict('records')
    }
    
    return info

def load_model_online(model_name="facebook/blenderbot-400M-distill", use_auth_token=None):
    """
    Load the model directly from Hugging Face Hub without caching it locally
    
    Args:
        model_name: Name of the model on Hugging Face
        use_auth_token: Optional Hugging Face auth token for private models
        
    Returns:
        model, tokenizer: The loaded model and tokenizer
    """
    print(f"Loading model from Hugging Face Hub: {model_name}")
    try:
        # Set force_download=True to always get the latest version
        # Set local_files_only=False to ensure it's fetched from online
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_auth_token=use_auth_token,
            local_files_only=False
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            use_auth_token=use_auth_token,
            local_files_only=False
        )
        
        print(f"Model {model_name} loaded successfully from online source")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from online: {e}")
        raise

def download_and_cache_model(model_name="facebook/blenderbot-400M-distill", cache_dir=None):
    """
    Download and cache the BlenderBot model and tokenizer
    
    Args:
        model_name: Name of the model on Hugging Face
        cache_dir: Directory to store the downloaded model
        
    Returns:
        model, tokenizer: The loaded model and tokenizer
    """
    print(f"Downloading and caching model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"Model {model_name} loaded successfully")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

if __name__ == "__main__":
    # This can be run as a standalone script to check the dataset
    try:
        # Load dataset from Hugging Face instead of local file
        df = load_and_preprocess_dataset()
        info = get_dataset_info(df)
        print(f"Dataset loaded successfully with {info['num_rows']} rows.")
        print(f"Columns: {info['columns']}")
        print("Sample data:")
        for i, row in enumerate(info['sample_data']):
            print(f"Row {i+1}: {row}")
        
        # Load the model from online
        print("\nLoading model from Hugging Face Hub...")
        model, tokenizer = load_model_online()
        print("Model loaded successfully!")
        
        # You can test the model with a simple input
        print("\nTesting model with a sample input...")
        inputs = tokenizer("Hello, how are you feeling today?", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Model response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")