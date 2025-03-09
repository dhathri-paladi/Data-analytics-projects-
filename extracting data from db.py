import mysql.connector
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Predefined skill mappings for normalization
skill_mapping = {
    "ML": "machine learning",
    "NLP": "natural language processing",
    "AI": "artificial intelligence",
    "DS": "data science"
}

# Function to normalize skills
def normalize_skills(skills):
    skills = skills.lower().split(", ")  # Split by comma and lowercase
    normalized = [skill_mapping.get(skill, skill) for skill in skills]
    return ", ".join(normalized)

# Custom tokenizer to handle multi-word terms
def custom_tokenizer(text):
    # Replace multi-word terms with underscores (e.g., "data science" -> "data_science")
    multi_word_terms = ["data science", "web development", "machine learning", "natural language processing", "cloud computing", ]
    for term in multi_word_terms:
        text = text.replace(term, term.replace(" ", "_"))
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Replace underscores with spaces
    tokens = [token.replace("_", " ") for token in tokens]
    return tokens

# Function to preprocess text columns
def preprocess_text_column(column):
    return column.apply(lambda text: [
        lemmatizer.lemmatize(word) for word in custom_tokenizer(text)
        if word.isalnum() or " " in word  # Allow multi-word terms
    ])

# Database connection details
HOST = "localhost"
USER = "root"
PASSWORD = "root"
DATABASE = "JobRecommendation"

try:
    # Connect to the MySQL database using mysql-connector
    connection = mysql.connector.connect(
        host=HOST,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )
    print("Connected to MySQL Database")

    # Query to extract data from each table
    queries = {
        "Users": "SELECT * FROM Users",
        "Jobs": "SELECT * FROM Jobs",
        "UserJobInteractions": "SELECT * FROM UserJobInteractions"
    }

    # Extract data into DataFrames
    data_frames = {}
    for table_name, query in queries.items():
        cursor = connection.cursor(dictionary=True)  # Use dictionary cursor for column names
        cursor.execute(query)
        data = cursor.fetchall()
        data_frames[table_name] = pd.DataFrame(data)
        print(f"Data extracted from table: {table_name}")
        cursor.close()

except mysql.connector.Error as e:
    print(f"Error: {e}")

finally:
    # Close the connection
    if connection:
        connection.close()
        print("MySQL connection is closed")

# Preprocess Users Table
if "Users" in data_frames:
    users_df = data_frames["Users"]
    users_df['skills'] = users_df['skills'].apply(normalize_skills)
    users_df['skills'] = preprocess_text_column(users_df['skills'])
    users_df['interests'] = preprocess_text_column(users_df['interests'])
    users_df.fillna({"preferred_location": "unknown"}, inplace=True)  # Handle missing locations

# Preprocess Jobs Table
if "Jobs" in data_frames:
    jobs_df = data_frames["Jobs"]
    jobs_df['skills_required'] = jobs_df['skills_required'].apply(normalize_skills)
    jobs_df['skills_required'] = preprocess_text_column(jobs_df['skills_required'])
    jobs_df['job_description'] = preprocess_text_column(jobs_df['job_description'])
    # Remove jobs older than a threshold (e.g., 1 year)
    jobs_df['posted_date'] = pd.to_datetime(jobs_df['posted_date'])
    jobs_df = jobs_df[jobs_df['posted_date'] >= datetime.now() - pd.Timedelta(days=365)]

# Preprocess User Job Interaction Table
if "UserJobInteractions" in data_frames:
    interactions_df = data_frames["UserJobInteractions"]
    interaction_type_mapping = {'Clicked': 1, 'Saved': 2, 'Applied': 3,'Liked': 4}
    interactions_df['interaction_score'] = interactions_df['interaction_type'].map(interaction_type_mapping)
    interactions_df.dropna(subset=['interaction_score'], inplace=True)  # Drop invalid interactions

# Save preprocessed data back to CSV (optional)
if "Users" in data_frames:
    users_df.to_csv("Preprocessed_Users.csv", index=False)
if "Jobs" in data_frames:
    jobs_df.to_csv("Preprocessed_Jobs.csv", index=False)
if "UserJobInteractions" in data_frames:
    interactions_df.to_csv("Preprocessed_UserJobInteractions.csv", index=False)

# Display preprocessed data
if "Users" in data_frames:
    print("\nPreprocessed Users Data:")
    print(users_df.head())
if "Jobs" in data_frames:
    print("\nPreprocessed Jobs Data:")
    print(jobs_df.head())
if "UserJobInteractions" in data_frames:
    print("\nPreprocessed User Job Interactions Data:")
    print(interactions_df.head())