import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json


def find_most_relevant_data(query, df, top_n=3):
    """
    Find the most relevant rows from the dataset based on the user query
    """
    # Combine all text columns in the dataset
    if 'question' in df.columns and 'answer' in df.columns:
        df['combined_text'] = df['question'] + ' ' + df['answer']
    elif 'text' in df.columns:
        df['combined_text'] = df['text']
    else:
        # Use all string columns
        text_columns = df.select_dtypes(include=['object']).columns
        df['combined_text'] = df[text_columns].apply(
            lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'].fillna(''))

    # Transform the query
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(
        query_vector, tfidf_matrix).flatten()

    # Get top N most similar rows
    most_similar_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    most_similar_rows = df.iloc[most_similar_indices]

    return most_similar_rows, cosine_similarities[most_similar_indices]

# Keep the original function for reference or if you need to switch back


def generate_response_with_model(query, df, model, tokenizer, max_length=100):
    """
    Generate a response based on the user query and the dataset
    """
    # Find relevant data from the dataset
    relevant_data, similarities = find_most_relevant_data(query, df)

    # Create a context from the relevant data
    if 'question' in df.columns and 'answer' in df.columns:
        context = "\n".join(
            [f"Q: {row['question']}\nA: {row['answer']}" for _, row in relevant_data.iterrows()])
    else:
        # Use all string columns
        text_columns = relevant_data.select_dtypes(include=['object']).columns
        context = "\n".join([f"{col}: {row[col]}" for _, row in relevant_data.iterrows(
        ) for col in text_columns if col != 'combined_text'])

    # Prepare input for the model
    prompt = f"Context information about mental health:\n{context}\n\nUser: {query}\nChatbot:"

    # Generate response using the model
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length + inputs.input_ids.shape[1],
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the chatbot's response
    if "Chatbot:" in response:
        response = response.split("Chatbot:")[1].strip()

    # Add a disclaimer for sensitive topics
    if any(keyword in query.lower() for keyword in ['suicide', 'kill', 'die', 'harm', 'hurt']):
        response += "\n\n*If you're experiencing thoughts of self-harm or suicide, please contact a mental health professional or Crisis helpline immediately. In the US, you can call the National Suicide Prevention Lifeline at 988 or 1-800-273-8255.*"

    return response


def generate_response(user_input, dataset, model_info, _):
    """
    Generate a response using the Llama 3 API via Groq

    Args:
        user_input: Preprocessed user input
        dataset: Mental health dataset
        model_info: Dictionary containing API key and model name
        _: Placeholder for tokenizer (not used with API)

    Returns:
        str: Generated response
    """
    api_key = model_info["api_key"]

    # Find relevant information from the dataset
    relevant_data, similarities = find_most_relevant_data(user_input, dataset)

    # Create a context from the relevant data
    if 'question' in dataset.columns and 'answer' in dataset.columns:
        context = "\n".join(
            [f"Q: {row['question']}\nA: {row['answer']}" for _, row in relevant_data.iterrows()])
    else:
        # Use all string columns
        text_columns = relevant_data.select_dtypes(include=['object']).columns
        context = "\n".join([f"{col}: {row[col]}" for _, row in relevant_data.iterrows(
        ) for col in text_columns if col != 'combined_text'])

    # Make API call to Groq for Llama 3
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Include relevant mental health information in the system prompt
    system_prompt = """You are a mental health support chatbot powered by Llama 3. 
    Be empathetic, supportive, and provide helpful resources when appropriate.
    Always maintain a compassionate tone and prioritize user safety.
    If the user expresses thoughts of self-harm or suicide, include Crisis resources in your response.
    Never claim to be a replacement for professional mental health care."""

    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is some relevant mental health information that might help with your response:\n{context}\n\nUser query: {user_input}"}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]

            # Add a disclaimer for sensitive topics
            if any(keyword in user_input.lower() for keyword in ['suicide', 'kill', 'die', 'harm', 'hurt']):
                response_text += "\n\n*If you're experiencing thoughts of self-harm or suicide, please contact a mental health professional or Crisis helpline immediately. In the US, you can call the National Suicide Prevention Lifeline at 988 or 1-800-273-8255.*"

            return response_text
        else:
            return f"Error: Failed to get response from Groq API. Status code: {response.status_code}. Details: {response.text}"

    except Exception as e:
        return f"Error: {str(e)}"
