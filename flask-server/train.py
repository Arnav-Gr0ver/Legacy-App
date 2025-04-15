import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_and_prepare_data(file_path='data/Disease_symptom_and_patient_profile_dataset.csv'):
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please provide the correct file path.")
        return None

def preprocess_data(df):
    if df is None:
        return None, None, None, None, None, None
    
    # Map Yes/No to 1/0 for binary features
    binary_map = {'Yes': 1, 'No': 0, 'Positive': 1, 'Negative': 0}
    for col in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Outcome Variable']:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
    
    # Define features and target
    X = df.drop(['Disease', 'Outcome Variable'], axis=1)
    y = df['Disease']
    
    # Get feature types for preprocessing
    categorical_features = ['Gender', 'Blood Pressure', 'Cholesterol Level']
    categorical_indices = [i for i, col in enumerate(X.columns) if col in categorical_features]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_indices)
        ],
        remainder='passthrough'
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create encoders for later use in prediction
    le_disease = LabelEncoder()
    le_disease.fit(df['Disease'])
    
    # Store column information for inference
    feature_names = X.columns.tolist()
    
    return X_train, X_test, y_train, y_test, preprocessor, le_disease, feature_names

def build_and_train_model(X_train, X_test, y_train, y_test, preprocessor):
    if X_train is None:
        return None, 0
    
    # Create and train the model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, pd.concat([X_train, X_test]), 
                               pd.concat([y_train, y_test]), cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    
    return model, accuracy

def save_model(model, le_disease, feature_names, file_path='disease_prediction_model.pkl'):
    if model is None:
        return
    
    # Save the model, encoders, and feature names
    with open(file_path, 'wb') as f:
        pickle.dump({
            'model': model, 
            'le_disease': le_disease,
            'feature_names': feature_names
        }, f)
    
    print(f"Model saved to {file_path}")

def inference_example(model, le_disease, example_data, feature_names):
    """
    Perform inference using the trained model.
    
    Args:
        model: Trained model
        le_disease: Label encoder for disease names
        example_data: Input data for prediction
        feature_names: List of feature names expected by the model
    
    Returns:
        Tuple of (predicted disease, probability dataframe)
    """
    if model is None:
        return None, None
    
    # Make sure example data has the expected format
    if isinstance(example_data, pd.DataFrame):
        # Check if columns match expected features
        if not all(col in feature_names for col in example_data.columns):
            print("Warning: Input data columns don't match model features")
            # Reorder columns to match expected features
            example_data = example_data.reindex(columns=feature_names)
    
    # Predict on example data
    prediction_indices = model.predict(example_data)
    prediction_proba = model.predict_proba(example_data)
    
    # Get the predicted disease
    disease = le_disease.inverse_transform(prediction_indices)
    
    # Get prediction probabilities for all diseases
    classes = le_disease.classes_
    proba_df = pd.DataFrame(prediction_proba, columns=classes)
    
    return disease, proba_df

def plot_feature_importance(model, X, feature_names):
    if model is None:
        return
    
    # Get feature names after preprocessing
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    # Extract feature names
    categorical_features = ['Gender', 'Blood Pressure', 'Cholesterol Level']
    numerical_features = [col for col in feature_names if col not in categorical_features]
    
    # Get one-hot encoded feature names
    cat_indices = [i for i, col in enumerate(feature_names) if col in categorical_features]
    if cat_indices:
        cat_encoder = preprocessor.transformers_[0][1]
        cat_cols = []
        for i, col in enumerate([feature_names[idx] for idx in cat_indices]):
            if i < len(cat_encoder.categories_):
                for cat in cat_encoder.categories_[i]:
                    cat_cols.append(f"{col}_{cat}")
            else:
                cat_cols.append(f"{col}_unknown")
    else:
        cat_cols = []
    
    processed_feature_names = cat_cols + numerical_features
    
    # Plot feature importance
    importances = classifier.feature_importances_
    
    # Make sure we have the right number of feature names
    if len(importances) != len(processed_feature_names):
        print(f"Warning: Number of features ({len(processed_feature_names)}) doesn't match importances ({len(importances)})")
        processed_feature_names = [f"Feature_{i}" for i in range(len(importances))]
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [processed_feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("Feature importance plot saved as 'feature_importance.png'")

def main():
    # Load and preprocess data
    df = load_and_prepare_data()
    X_train, X_test, y_train, y_test, preprocessor, le_disease, feature_names = preprocess_data(df)
    
    # Build and train model
    model, accuracy = build_and_train_model(X_train, X_test, y_train, y_test, preprocessor)
    
    # Save the model
    save_model(model, le_disease, feature_names)
    
    # Plot feature importance
    if model is not None:
        plot_feature_importance(model, pd.concat([X_train, X_test]), feature_names)
    
    # Example inference
    if X_test is not None and model is not None:
        print("\nExample Inference:")
        example = X_test.iloc[0:1]  # Use the first test example
        
        try:
            disease, proba = inference_example(model, le_disease, example, feature_names)
            print(f"Input features: {example.values}")
            print(f"Predicted disease: {disease[0]}")
            print("Prediction probabilities:")
            print(proba.iloc[0].sort_values(ascending=False).head())
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            print("This could be due to a mismatch between training labels and prediction outputs.")
            print("Consider using model.predict_proba directly to check raw class probabilities.")
            
            # Alternative approach: just show raw probabilities
            raw_proba = model.predict_proba(example)
            print("\nRaw prediction probabilities:")
            print(raw_proba)
            
            # Get the most likely class index
            most_likely_idx = np.argmax(raw_proba, axis=1)[0]
            print(f"Most likely class index: {most_likely_idx}")
            
            # If the index is valid, show the class name
            if 0 <= most_likely_idx < len(le_disease.classes_):
                print(f"Most likely disease: {le_disease.classes_[most_likely_idx]}")
            else:
                print("Index out of range for disease labels")

if __name__ == "__main__":
    main()