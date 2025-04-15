from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from scipy import spatial
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time
import joblib
import pandas as pd
import os
import re

app = Flask(
    __name__,
    static_folder='../client/build/static',
    template_folder='../client/build'
)
CORS(app)

# Hugging Face Credentials
hf_username = os.environ['HF_USERNAME']
hf_token = os.environ['HF_TOKEN']

# Load LLaMA Model and Tokenizer
llama_model_id = "meta-llama/Llama-3.2-1B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id, use_auth_token=hf_token)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_id, use_auth_token=hf_token)

# Load preprocessor and predictive model
try:
    preprocessor = joblib.load('models/preprocessor.pkl')
    predictive_model = joblib.load('models/predictive_model.pkl')
    le_target = joblib.load('models/label_encoder.pkl')
    print("Predictive models loaded successfully")
except Exception as e:
    print(f"Error loading predictive models: {e}")
    # Create placeholders to prevent application crash
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    preprocessor = StandardScaler()
    le_target = LabelEncoder()
    
# Database connection
def get_db_connection():
    conn = sqlite3.connect('legacy_search.db')
    conn.row_factory = sqlite3.Row
    return conn

def is_mental_health_related(query):
    """Check if the query is related to mental health or clinical topics."""
    prompt = f"""
    Determine if the following query is related to mental health, psychology, or clinical topics.
    Return only 'YES' if it is related or 'NO' if it is not related.
    
    Query: {query}
    """
    
    inputs = llama_tokenizer(prompt, return_tensors="pt")
    outputs = llama_model.generate(**inputs, max_new_tokens=10)
    response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return "YES" in response.upper()

def generate_llm_response(query):
    """Generate a response using the LLM for general mental health questions."""
    prompt = f"""
    You are a mental health information assistant. Provide helpful, accurate information about the following mental health question.
    Remember to be empathetic, informative, and evidence-based in your response.
    
    Question: {query}
    
    Answer:
    """
    
    inputs = llama_tokenizer(prompt, return_tensors="pt")
    outputs = llama_model.generate(**inputs, max_new_tokens=500)
    response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the answer part
    try:
        answer = response.split("Answer:")[1].strip()
    except:
        answer = response
        
    return answer

def semantic_search(query, top_n=1):
    """Find relevant examples from the database using semantic search."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get embedding for the query
    inputs = llama_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = llama_model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    
    cursor.execute('SELECT id, context, response, context_embedding FROM conversations')
    rows = cursor.fetchall()
    
    similarities = []
    for row in rows:
        id, context, response, context_embedding_bytes = row
        context_embedding = np.frombuffer(context_embedding_bytes, dtype=np.float32)
        similarity = 1 - spatial.distance.cosine(query_embedding, context_embedding)
        similarities.append({
            "id": id,
            "context": context,
            "response": response,
            "similarity": float(similarity),
            "source": "Clinical Database",
            "date": "2024"
        })
    
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = similarities[:top_n]
    
    conn.close()
    return top_results

def generate_predictive_response(query):
    """Extract symptoms from the query and make a prediction using the trained model."""
    try:
        # Extract symptoms from query
        symptoms = {
            'Fever': 'Yes' if 'fever' in query.lower() else 'No',
            'Cough': 'Yes' if 'cough' in query.lower() else 'No',
            'Fatigue': 'Yes' if 'fatigue' in query.lower() or 'tired' in query.lower() else 'No',
            'Difficulty Breathing': 'Yes' if any(term in query.lower() for term in ['breathing', 'breath', 'shortness']) else 'No',
            'Gender': 'Male' if 'male' in query.lower() else 'Female' if 'female' in query.lower() else 'Unknown',
            'Blood Pressure': 'High' if 'high blood pressure' in query.lower() else 'Low' if 'low blood pressure' in query.lower() else 'Normal',
            'Cholesterol Level': 'High' if 'high cholesterol' in query.lower() else 'Low' if 'low cholesterol' in query.lower() else 'Normal',
            'Age': int(''.join(filter(str.isdigit, query))) if any(char.isdigit() for char in query) else 30
        }
        
        # Load the model components from the saved file
        model_data = joblib.load('disease_prediction_model.pkl')
        model = model_data['model']
        le_disease = model_data['le_disease']
        feature_names = model_data.get('feature_names', None)
        
        # Map binary values to 1/0
        binary_map = {'Yes': 1, 'No': 0}
        for key in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']:
            if key in symptoms:
                symptoms[key] = binary_map.get(symptoms[key], 0)
        
        # Create a DataFrame with the extracted symptoms
        input_data = pd.DataFrame([symptoms])
        
        # Ensure columns match expected features if feature_names was saved
        if feature_names:
            for col in feature_names:
                if col not in input_data.columns:
                    input_data[col] = 0  # Default value for missing features
            # Reorder columns to match the expected order
            input_data = input_data[feature_names]
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Get the top 3 most likely diseases with their probabilities
        top_indices = prediction_proba[0].argsort()[-3:][::-1]
        top_diseases = le_disease.inverse_transform(top_indices)
        top_probabilities = prediction_proba[0][top_indices]
        
        # Construct response
        response = f"<p><strong>Predicted Disease:</strong> {top_diseases[0]} (Confidence: {top_probabilities[0]*100:.2f}%)</p>"
        
        # Add other likely diseases
        response += "<p><strong>Other Possibilities:</strong></p><ul>"
        for disease, prob in zip(top_diseases[1:], top_probabilities[1:]):
            response += f"<li>{disease} (Confidence: {prob*100:.2f}%)</li>"
        response += "</ul>"
        
        # Add symptom analysis
        response += "<p><strong>Based on these symptoms:</strong></p><ul>"
        for symptom, value in symptoms.items():
            display_value = value
            if symptom in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']:
                display_value = 'Yes' if value == 1 else 'No'
            response += f"<li>{symptom}: {display_value}</li>"
        response += "</ul>"
        
        response += "<p><em>Note: This is a prediction based on machine learning and should not replace professional medical advice.</em></p>"
        
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in disease prediction: {error_details}")
        return f"<p>Error making prediction: {str(e)}</p><p>Please consult a healthcare professional for accurate diagnosis.</p>"

def determine_query_type(query):
    """Determine if the query needs examples, prediction, or general info."""
    prompt = f"""Classify the following query into one of these categories: GENERAL, PREDICTION, CASE.

Query: "{query}"

Return the label only in this format: ANSWER:<LABEL>, where <LABEL> is GENERAL, PREDICTION, or CASE.
Answer:"""
    
    inputs = llama_tokenizer(prompt, return_tensors="pt")
    outputs = llama_model.generate(**inputs, max_new_tokens=10)
    response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()

    # Try to extract the label from the format ANSWER:<LABEL>
    match = re.search(r'ANSWER\s*:\s*(GENERAL|PREDICTION|CASE)', response)
    if match:
        label = match.group(1).lower()
    else:
        # fallback: loose keyword search
        if "PREDICTION" in response:
            label = "prediction"
        elif "CASE" in response:
            label = "examples"
        else:
            label = "general"

    return label


def generate_response(query):
    """Main function to generate appropriate response based on query type."""
    if not is_mental_health_related(query):
        return "I can only answer questions related to mental health and clinical topics.", []
    
    query_type = determine_query_type(query)

    if query_type == "prediction":
        response = generate_predictive_response(query)
        return response, []
    
    elif query_type == "examples":
        contexts = semantic_search(query)
        if contexts:
            # Combine the examples into a meaningful response
            examples_html = "<p>Here are some relevant examples:</p>"
            for i, context in enumerate(contexts[:3], 1):
                examples_html += f"<div class='example'><h4>Example {i}</h4><p>{context['response']}</p></div>"
            
            return examples_html, contexts
        else:
            return generate_llm_response(query), []
    
    else:  # general
        return generate_llm_response(query), []

@app.route('/api/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        start_time = time.time()
        response, contexts = generate_response(query)
        title = f"Results for '{query}'"
        title = title[:50] + "..." if len(title) > 50 else title
        
        return jsonify({
            "title": title,
            "response": response,
            "contexts": contexts,
            "query": query,
            "response_time": round(time.time() - start_time, 2)
        })
    
    except Exception as e:
        return jsonify({
            "error": "Processing error",
            "details": str(e)
        }), 500

@app.route('/', methods=['GET'])
def home():
    return "Mental Health Search API - Semantic Retrieval & Predictive Model"

if __name__ == '__main__':
    # Check if database exists, if not create a simple one for demo
    if not os.path.exists('legacy_search.db'):
        print("Creating sample database...")
        conn = sqlite3.connect('legacy_search.db')
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            context TEXT,
            response TEXT,
            context_embedding BLOB
        )
        ''')
        conn.commit()
        conn.close()
        
    print("Starting Mental Health Search server...")
    app.run()