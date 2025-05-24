import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources - make this more robust
def ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded"""
    resources = [
        'punkt',
        'stopwords',
        'wordnet'
    ]
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)

# Call this function at module import time
ensure_nltk_resources()

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing special characters,
    removing stopwords, and lemmatizing
    
    Args:
        text (str): The text to preprocess
        
    Returns:
        str: The preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        # If tokenization fails, try downloading resources again and retry
        ensure_nltk_resources()
        tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text