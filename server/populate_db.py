import pandas as pd
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy import spatial

# Load the dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Generate embeddings for contexts and responses
def generate_embeddings(model, texts):
    return model.encode(texts, batch_size=32, show_progress_bar=True)

# Create SQLite database and table
def create_database(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            context TEXT,
            response TEXT,
            context_embedding BLOB,
            response_embedding BLOB
        )
    ''')
    return conn, cursor

# Populate the database with embeddings
def populate_database(conn, cursor, contexts, responses, context_embeddings, response_embeddings):
    for context, response, context_embedding, response_embedding in zip(contexts, responses, context_embeddings, response_embeddings):
        context_embedding_bytes = np.array(context_embedding).tobytes()
        response_embedding_bytes = np.array(response_embedding).tobytes()
        cursor.execute('''
            INSERT INTO conversations (context, response, context_embedding, response_embedding)
            VALUES (?, ?, ?, ?)
        ''', (context, response, context_embedding_bytes, response_embedding_bytes))
    conn.commit()

# Main function
def main():
    dataset_file = 'train.csv'
    db_name = 'mental_health.db'
    
    # Load dataset
    df = load_dataset(dataset_file)
    contexts = df['Context'].tolist()
    responses = df['Response'].tolist()
    
    # Initialize sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    context_embeddings = generate_embeddings(model, contexts)
    response_embeddings = generate_embeddings(model, responses)
    
    # Create and populate database
    conn, cursor = create_database(db_name)
    populate_database(conn, cursor, contexts, responses, context_embeddings, response_embeddings)
    conn.close()

if __name__ == '__main__':
    main()
