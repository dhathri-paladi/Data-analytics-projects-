from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast
import mysql.connector

# Database Connection Details
HOST = "localhost"
USER = "root"
PASSWORD = "root"
DATABASE = "JobRecommendation"

# Load preprocessed data
users_df = pd.read_csv(r"C:\Users\SREE GANESHA\Desktop\chandana1\Preprocessed_Users.csv")
jobs_df = pd.read_csv(r"C:\Users\SREE GANESHA\Desktop\chandana1\Preprocessed_Jobs.csv")

# Function to safely parse lists while preserving multi-word phrases
def parse_list(column_value):
    if pd.notnull(column_value):
        try:
            parsed_list = ast.literal_eval(column_value)
            if isinstance(parsed_list, list):
                return " ".join(parsed_list)
        except (SyntaxError, ValueError):
            return column_value.replace(",", " ")
    return ""

# Apply parsing function
users_df['profile'] = users_df.apply(lambda x: parse_list(x['skills']) + " " + parse_list(x['interests']), axis=1)
jobs_df['job_profile'] = jobs_df.apply(lambda x: parse_list(x['skills_required']) + " " + parse_list(x['job_description']), axis=1)

# Remove rows with empty profiles or job descriptions
users_df = users_df[users_df['profile'].str.strip() != ""]
jobs_df = jobs_df[jobs_df['job_profile'].str.strip() != ""]

# Step 1: Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform user profiles and job profiles
tfidf_users = tfidf_vectorizer.fit_transform(users_df['profile'])
tfidf_jobs = tfidf_vectorizer.transform(jobs_df['job_profile'])

# Step 2: Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_users, tfidf_jobs)

# Connect to MySQL database
def connect_db():
    return mysql.connector.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)

# Function to store recommendations in the database
def store_recommendations(user_id, recommendations):
    try:
        connection = connect_db()
        cursor = connection.cursor()

        # Create table if not exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS JobRecommendations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            job_title VARCHAR(255),
            company_name VARCHAR(255),
            skills_required TEXT,
            job_location VARCHAR(255),
            similarity_score FLOAT
        )
        """)

        # Insert recommendations into the database
        for job in recommendations:
            query = """
            INSERT INTO JobRecommendations (user_id, job_title, company_name, skills_required, job_location, similarity_score) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                int(user_id),  # Ensure user_id is an integer
                job['job_title'],
                job['company_name'],
                job['skills_required'],
                job['job_location'],
                float(job['similarity_score'])  # Ensure similarity_score is a float
            ))

        connection.commit()
        print(f"Recommendations for User {user_id} stored successfully in the database.")

    except mysql.connector.Error as e:
        print(f"Database Error: {e}")

    finally:
        if connection:
            cursor.close()
            connection.close()

# Function to get recommendations for a specific user
def get_recommendations(user_id):
    try:
        connection = connect_db()
        cursor = connection.cursor()

        query = """
        SELECT job_title, company_name, skills_required, job_location, similarity_score 
        FROM JobRecommendations 
        WHERE user_id = %s 
        ORDER BY similarity_score DESC
        """
        cursor.execute(query, (int(user_id),))  # Ensure user_id is an integer
        recommendations = cursor.fetchall()

        if recommendations:
            print(f"\nJob Recommendations for User {user_id}:")
            for rec in recommendations:
                print(f"Job Title: {rec[0]}, Company: {rec[1]}, Skills: {rec[2]}, Location: {rec[3]}, Score: {rec[4]:.2f}")
        else:
            print(f"No recommendations found for User {user_id}.")

    except mysql.connector.Error as e:
        print(f"Database Error: {e}")

    finally:
        if connection:
            cursor.close()
            connection.close()

# Step 3: Compute and store recommendations for all users
for user_idx, user_similarities in enumerate(cosine_sim):
    user_id = int(users_df.iloc[user_idx]['user_id'])  # Convert user_id to int
    preferred_location = users_df.iloc[user_idx]['preferred_location']
    
    # Add a location match flag to prioritize jobs in the user's preferred location
    jobs_df['location_match'] = jobs_df['job_location'] == preferred_location
    
    # Create a DataFrame of job similarity scores
    job_recommendations = jobs_df.copy()
    job_recommendations['similarity_score'] = user_similarities.astype(float)  # Convert similarity scores to float
    job_recommendations = job_recommendations.sort_values(
        by=['location_match', 'similarity_score'], ascending=[False, False]
    )
    
    # Select the top 5 job recommendations
    top_jobs = job_recommendations.head(5)[['job_title', 'company_name', 'skills_required', 'job_location', 'similarity_score']]
    
    # Convert to list of dictionaries
    top_jobs_list = top_jobs.to_dict(orient='records')

    # Store recommendations in the database
    store_recommendations(user_id, top_jobs_list)

# Step 4: Allow user to fetch recommendations by entering user_id
if __name__ == "__main__":
    user_id_input = int(input("Enter User ID to get job recommendations: "))
    get_recommendations(user_id_input)