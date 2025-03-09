import pandas as pd
import numpy as np
import mysql.connector
from sklearn.decomposition import TruncatedSVD

# Database Connection Details
HOST = "localhost"
USER = "root"
PASSWORD = "root"
DATABASE = "JobRecommendation"

# Load the Data
interactions_df = pd.read_csv(r"C:\Users\SREE GANESHA\Desktop\chandana1\Preprocessed_UserJobInteractions.csv")
jobs_df = pd.read_csv(r"C:\Users\SREE GANESHA\Desktop\chandana1\Preprocessed_Jobs.csv")

# Ensure job_id is treated as an integer
jobs_df['job_id'] = jobs_df['job_id'].astype(int, errors='ignore')
interactions_df['job_id'] = interactions_df['job_id'].astype(int, errors='ignore')

# Ensure no missing job titles
jobs_df.fillna({"job_title": "Unknown Title", "job_description": "No description available"}, inplace=True)

# Convert Interaction Type to Scores
interaction_type_mapping = {'Clicked': 1, 'Saved': 2, 'Liked': 3, 'Applied': 4}
interactions_df['interaction_score'] = interactions_df['interaction_type'].map(interaction_type_mapping)

# Create User-Job Interaction Matrix
interaction_matrix = interactions_df.pivot_table(
    index='user_id',
    columns='job_id',
    values='interaction_score',
    aggfunc='sum',
    fill_value=0
)

# Apply SVD
n_components = min(interaction_matrix.shape[1], 50)
svd = TruncatedSVD(n_components=n_components)
matrix_factorized = svd.fit_transform(interaction_matrix)
reconstructed_matrix = np.dot(matrix_factorized, svd.components_)

# Database Connection
def connect_db():
    return mysql.connector.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)

# Create Table for Recommendations
def create_table():
    try:
        connection = connect_db()
        cursor = connection.cursor()

        query = """
        CREATE TABLE IF NOT EXISTS JobRecommendations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            job_id INT NOT NULL,
            job_title VARCHAR(255),
            job_description TEXT,
            predicted_score FLOAT,
            UNIQUE KEY unique_recommendation (user_id, job_id)
        );
        """
        cursor.execute(query)
        connection.commit()

    except mysql.connector.Error as e:
        print(f"❌ Database Error: {e}")

    finally:
        cursor.close()
        connection.close()

# Get Job Recommendations for a User
def recommend_jobs_for_user(user_id, top_n=5):
    if user_id not in interaction_matrix.index:
        raise ValueError(f"User ID {user_id} not found in dataset.")

    user_index = interaction_matrix.index.get_loc(user_id)
    user_predictions = reconstructed_matrix[user_index]

    interacted_jobs = interactions_df[interactions_df['user_id'] == user_id]['job_id'].tolist()

    recommended_jobs = pd.Series(user_predictions, index=interaction_matrix.columns)
    recommended_jobs = recommended_jobs[~recommended_jobs.index.isin(interacted_jobs)]

    top_job_ids = recommended_jobs.sort_values(ascending=False).head(top_n).index.tolist()

    # Merge with jobs_df and ensure correct job titles
    recommended_jobs_details = jobs_df[jobs_df['job_id'].isin(top_job_ids)].set_index('job_id').reindex(top_job_ids).reset_index()

    # Ensure job titles are correctly filled
    recommended_jobs_details["job_title"] = recommended_jobs_details["job_title"].fillna("Unknown Title")
    recommended_jobs_details["job_description"] = recommended_jobs_details["job_description"].fillna("No description available")

    # Merge predicted scores correctly
    recommended_jobs_details["predicted_score"] = recommended_jobs.reindex(top_job_ids, fill_value=0).values

    return recommended_jobs_details[['job_id', 'job_title', 'job_description', 'predicted_score']]

# Store Recommendations in MySQL
def store_recommendations(user_id, recommendations):
    try:
        connection = connect_db()
        cursor = connection.cursor()

        create_table()

        for _, job in recommendations.iterrows():
            job_id = int(job['job_id'])
            job_title = job['job_title'] if pd.notna(job['job_title']) else "Unknown Title"
            job_description = job['job_description'] if pd.notna(job['job_description']) else "No description available"
            predicted_score = float(job['predicted_score']) if pd.notna(job['predicted_score']) else 0.0
            
            query = """
            INSERT INTO JobRecommendations (user_id, job_id, job_title, job_description, predicted_score)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE predicted_score = VALUES(predicted_score)
            """
            cursor.execute(query, (user_id, job_id, job_title, job_description, predicted_score))

        connection.commit()
        print(f"✅ Stored recommendations for User {user_id}")

    except mysql.connector.Error as e:
        print(f"❌ Database Error: {e}")

    finally:
        cursor.close()
        connection.close()

# Retrieve Recommendations from MySQL
def get_recommendations(user_id):
    try:
        connection = connect_db()
        cursor = connection.cursor()

        query = """
        SELECT job_id, job_title, job_description, predicted_score 
        FROM JobRecommendations 
        WHERE user_id = %s 
        ORDER BY predicted_score DESC
        """
        cursor.execute(query, (user_id,))
        recommendations = cursor.fetchall()

        if recommendations:
            print(f"\n🔹 Job Recommendations for User {user_id}:")
            for rec in recommendations:
                # Replace None with 0.00 if predicted_score is None
                predicted_score = rec[3] if rec[3] is not None else 0.00
                print(f"📌 Job ID: {rec[0]}, Title: {rec[1]}, Score: {predicted_score:.2f}")
        else:
            print(f"❌ No recommendations found for User {user_id}.")

    except mysql.connector.Error as e:
        print(f"❌ Database Error: {e}")

    finally:
        cursor.close()
        connection.close()

# Compute and Store Recommendations
create_table()

for user_id in interaction_matrix.index:
    try:
        top_jobs = recommend_jobs_for_user(user_id, top_n=5)
        top_jobs['user_id'] = user_id
        store_recommendations(user_id, top_jobs)
    except Exception as e:
        print(f"❌ Error processing user {user_id}: {e}")

# Get User Recommendations on Demand
if __name__ == "__main__":
    while True:
        try:
            user_id_input = input("Enter User ID to get recommendations (or 'exit' to quit): ").strip()
            if user_id_input.lower() == 'exit':
                break
                
            user_id_input = int(user_id_input)
            get_recommendations(user_id_input)

        except ValueError:
            print("❌ Invalid input! Please enter a valid User ID (integer).")
