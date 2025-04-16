from flask import Flask, request, jsonify
import os
import pandas as pd
import joblib
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

app = Flask(
    __name__,
    static_folder='../client/build',
    template_folder='../client/build', 
    static_url_path='/'
)

# Load environment variables
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
if not OpenAI.api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=OpenAI.api_key)

# Load case studies data
try:
    case_studies_df = pd.read_csv('data/conversations.csv')
    # Create a TF-IDF vectorizer for case matching
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(case_studies_df['Context'])
    print(f"Loaded {len(case_studies_df)} case studies")
except Exception as e:
    print(f"Error loading case studies: {str(e)}")
    case_studies_df = pd.DataFrame(columns=['Context', 'Response'])
    tfidf_vectorizer = None
    tfidf_matrix = None

def classify_query(query):
    """
    Classifies medical queries into 'PREDICTION', 'CASE', or 'GENERAL' categories
    using OpenAI instead of regex patterns.
    """
    from openai import OpenAI
    
    client = OpenAI()  # This assumes API key is set in environment variable
    
    # Create a prompt that explains the classification task
    prompt = f"""
    Classify the following medical query into exactly one of these categories:
    - PREDICTION: Queries asking for predictions about medical outcomes, risks, or probabilities
    - CASE: Queries describing specific patient cases, examples, or medical histories
    - GENERAL: General medical questions that don't fit the above categories
    
    Examples:
    "What is the likelihood of recovery for a 45-year-old male with pneumonia?" → PREDICTION
    "Patient with history of diabetes and hypertension presenting with chest pain" → CASE
    "What medications are commonly prescribed for hypertension?" → GENERAL
    
    Query to classify: {query}
    
    Classification (return only PREDICTION, CASE, or GENERAL):
    """
    
    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",  # You can use other models as needed
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # Use low temperature for consistent results
        max_tokens=10  # We only need a short response
    )
    
    # Extract and return the classification
    classification = response.choices[0].message.content.strip()
    
    # Ensure the response is one of our expected categories
    valid_categories = ["PREDICTION", "CASE", "GENERAL"]
    if classification not in valid_categories:
        # Fall back to a default if the API returns an unexpected value
        return "GENERAL"
    
    return classification

# Function to generate predictive response
def generate_predictive_response(query):
    try:
        # Symptom Extraction
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

        # Model Loading and Prediction
        try:
            model_data = joblib.load('disease_prediction_model.pkl')
            model = model_data['model']
            le_disease = model_data['le_disease']
            feature_names = model_data.get('feature_names', None)
        except Exception as model_error:
            # If model can't be loaded, simulate a response
            return f"""
                <p><strong>Note:</strong> Could not load prediction model: {str(model_error)}</p>
                <p><strong>Extracted symptoms:</strong></p>
                <ul>
                    {''.join(f'<li>{symptom}: {value}</li>' for symptom, value in symptoms.items())}
                </ul>
                <p><em>Please ensure the model file exists and is accessible.</em></p>
            """

        # Feature Preparation
        binary_map = {'Yes': 1, 'No': 0}
        for key in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']:
            symptoms[key] = binary_map.get(symptoms[key], 0)

        input_data = pd.DataFrame([symptoms])
        if feature_names:
            for col in feature_names:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[feature_names]

        # Prediction and Formatting
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        top_indices = prediction_proba[0].argsort()[-3:][::-1]
        top_diseases = le_disease.inverse_transform(top_indices)
        top_probabilities = prediction_proba[0][top_indices]

        response = f"""
            <p><strong>Predicted Condition:</strong> {top_diseases[0]} (Confidence: {top_probabilities[0]*100:.2f}%)</p>
            <p><strong>Other Possibilities:</strong></p><ul>
            """
        for disease, prob in zip(top_diseases[1:], top_probabilities[1:]):
            response += f"<li>{disease} (Confidence: {prob*100:.2f}%)</li>"
        response += "</ul>"

        response += """
            <p><strong>Based on these symptoms:</strong></p><ul>
            """
        for symptom, value in symptoms.items():
            display_value = 'Yes' if value == 1 else 'No' if symptom in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing'] else value
            response += f"<li>{symptom}: {display_value}</li>"
        response += """
            </ul>
            <p><em>Note: This prediction is for informational purposes and should not replace professional medical advice.</em></p>
            """
        return response
    except Exception as e:
        return f"<p>Error: {str(e)}</p><p>Please consult a healthcare professional for accurate diagnosis.</p>"

# Function to find similar case studies
def find_similar_cases(query, n=3):
    if tfidf_vectorizer is None or tfidf_matrix is None or len(case_studies_df) == 0:
        return []

    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-n:][::-1]

    results = []
    for idx in top_indices:
        if similarity_scores[idx] > 0.1:  # Minimum similarity threshold
            results.append({
                'context': case_studies_df.iloc[idx]['Context'],
                'response': case_studies_df.iloc[idx]['Response'],
                'similarity': float(similarity_scores[idx])
            })

    return results

# Generate response using OpenAI's API
def generate_ai_response(query, query_type):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful mental health counseling assistant. Provide evidence-based, professional guidance to counselors on how to best help their patients."},
            {"role": "user", "content": query}
        ]

        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=messages,
            max_tokens=1024,
            temperature=0.7
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    query_type = classify_query(query)

    print(query_type)

    response = {
        "query_type": query_type,
        "content": "",
        "similar_cases": []
    }

    if query_type == 'PREDICTION':
        response["content"] = generate_predictive_response(query)
    elif query_type == 'CASE':
        similar_cases = find_similar_cases(query)
        response["similar_cases"] = similar_cases

        if similar_cases:
            # Generate additional insights based on the cases
            case_insights_prompt = f"Based on the following similar cases, provide insights for the counselor:\n\n"
            for i, case in enumerate(similar_cases):
                case_insights_prompt += f"Case {i+1}:\nContext: {case['context']}\nResponse: {case['response']}\n\n"
            case_insights_prompt += f"\nThe counselor's query was: {query}\nWhat insights can you provide to help with this case?"

            response["content"] = generate_ai_response(case_insights_prompt, query_type)
        else:
            response["content"] = generate_ai_response(f"I couldn't find similar cases, but here's my advice for: {query}", query_type)
    else:  # GENERAL
        response["content"] = generate_ai_response(query, query_type)

    return jsonify(response)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return app.send_static_file(path)
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))