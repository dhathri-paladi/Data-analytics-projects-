import pandas as pd
import numpy as np
import mysql.connector
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Database Connection Details
HOST = "localhost"
USER = "root"
PASSWORD = "root"
DATABASE = "JobRecommendation"

def connect_db():
    """Establish a connection to the MySQL database."""
    return mysql.connector.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)

# Load Data (Fixed File Paths)
users_df = pd.read_csv(r"C:\Users\SREE GANESHA\Desktop\chandana1\Preprocessed_Users.csv")
jobs_df = pd.read_csv(r"C:\Users\SREE GANESHA\Desktop\chandana1\Preprocessed_Jobs.csv")
interactions_df = pd.read_csv(r"C:\Users\SREE GANESHA\Desktop\chandana1\Preprocessed_UserJobInteractions.csv")

# Ensure required columns exist
print("Columns in jobs_df:", jobs_df.columns)
print("Columns in users_df:", users_df.columns)
print("Columns in interactions_df:", interactions_df.columns)

# Content-Based Filtering (TF-IDF + Cosine Similarity)
vectorizer = TfidfVectorizer()
tfidf_users = vectorizer.fit_transform(users_df['skills'] + " " + users_df['interests'])
tfidf_jobs = vectorizer.transform(jobs_df['skills_required'] + " " + jobs_df['job_description'])
content_sim = cosine_similarity(tfidf_users, tfidf_jobs)

# Collaborative Filtering (SVD for Matrix Factorization)
interaction_matrix = interactions_df.pivot_table(index='user_id', columns='job_id', values='interaction_score', aggfunc='sum', fill_value=0)
n_components = min(interaction_matrix.shape[1], 50) if interaction_matrix.shape[1] >= 50 else interaction_matrix.shape[1]
svd = TruncatedSVD(n_components=n_components)
matrix_factorized = svd.fit_transform(interaction_matrix)
collab_sim = np.dot(matrix_factorized, svd.components_)

# Hybrid Recommendation Function
def hybrid_recommendations(user_id, weight_content=0.5, weight_collab=0.5, top_n=5):
    """Generate hybrid job recommendations by combining content-based and collaborative filtering."""
    if user_id not in interaction_matrix.index:
        print(f"User {user_id} not found.")
        return None

    user_index = interaction_matrix.index.get_loc(user_id)
    content_scores = content_sim[user_index]

    # Check if user index is valid in collab_sim
    if user_index < collab_sim.shape[0]:
        collab_scores = collab_sim[user_index]
    else:
        collab_scores = np.zeros_like(content_scores)  # Fill missing values with zeros

    # Ensure dimensions match
    min_len = min(len(content_scores), len(collab_scores))
    content_scores = content_scores[:min_len]
    collab_scores = collab_scores[:min_len]

    hybrid_scores = (weight_content * content_scores) + (weight_collab * collab_scores)

    job_recommendations = pd.DataFrame({'job_id': jobs_df['job_id'][:min_len], 'hybrid_score': hybrid_scores})
    job_recommendations = job_recommendations.sort_values(by='hybrid_score', ascending=False).head(top_n)

    return job_recommendations.merge(jobs_df, on='job_id')[['job_id', 'job_title', 'job_description', 'hybrid_score']]

# Store Recommendations in MySQL
def store_recommendations(user_id, recommendations):
    """Store hybrid recommendations for a user in the database."""
    if recommendations is None or recommendations.empty:
        print(f"No recommendations for User {user_id}.")
        return

    try:
        connection = connect_db()
        cursor = connection.cursor()

        # Create table if not exists
        cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS HybridJobRecommendations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            job_id INT,
            job_title VARCHAR(255),
            job_description TEXT,
            final_score FLOAT,
            method VARCHAR(50)
        )""")

        # Insert recommendations
        for _, job in recommendations.iterrows():
            cursor.execute(""" 
            INSERT INTO HybridJobRecommendations (user_id, job_id, job_title, job_description, final_score, method) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """, (user_id, job['job_id'], job['job_title'], job['job_description'], job['hybrid_score'], 'Hybrid'))

        connection.commit()
        print(f"Recommendations for User {user_id} stored successfully.")
    
    except mysql.connector.Error as e:
        print(f"Database Error: {e}")
    
    finally:
        cursor.close()
        connection.close()

# Fetch Recommendations for a Specific User
def get_recommendations(user_id):
    """Retrieve stored recommendations for a specific user."""
    try:
        connection = connect_db()
        cursor = connection.cursor()
        cursor.execute(""" 
        SELECT job_id, job_title, job_description, final_score, method
        FROM HybridJobRecommendations 
        WHERE user_id = %s 
        ORDER BY final_score DESC
        """, (user_id,))
        
        recommendations = cursor.fetchall()
        if recommendations:
            print(f"\nRecommended Jobs for User {user_id}:")
            for rec in recommendations:
                print(f"Job ID: {rec[0]}, Title: {rec[1]}, Score: {rec[3]:.2f}, Method: {rec[4]}")
        else:
            print(f"No stored recommendations found for User {user_id}.")
    
    except mysql.connector.Error as e:
        print(f"Database Error: {e}")
    
    finally:
        cursor.close()
        connection.close()

# Generate and Store Recommendations for All Users
for user_id in interaction_matrix.index:
    recommendations = hybrid_recommendations(user_id)
    store_recommendations(user_id, recommendations)

# Handle User Input for Recommendation Retrieval
if __name__ == "__main__":
    user_input = input("Enter User ID: ").strip()
    if user_input.isdigit():  
        user_id_input = int(user_input)
        get_recommendations(user_id_input)
    else:
        print("Invalid input! Please enter a valid numeric User ID.")
