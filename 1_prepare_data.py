# Filename: 1_prepare_final_data.py
# Description: This single script handles all data loading, merging, cleaning,
# feature engineering, and preprocessing to create a model-ready dataset.
# FIX: Corrected the data loading logic to prevent columns from being swapped.

import pandas as pd
import requests
import time
import re
import os
from sklearn.preprocessing import MinMaxScaler

# --- 1. CONFIGURATION ---
API_KEY = 'YOUR_API_KEY_HERE'
MOVIELENS_PATH = 'C:\\Users\\User\\OneDrive\\Documents\\datathon\\recom_system_making\\data\\raw_data\\ml-1m\\'
FINAL_OUTPUT_FILE = 'preprocessed_movie_data.csv'

# --- 2. DATA LOADING & MERGING ---
print("Step 1: Loading and Merging MovieLens Data...")
try:
    # --- FIX STARTS HERE ---
    # Define column names in the exact order they appear in the .dat files
    r_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    u_names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
    m_names = ['MovieID', 'Title', 'Genres']

    # Load the full files with correct names, without using the ambiguous `usecols`
    ratings = pd.read_csv(os.path.join(MOVIELENS_PATH, 'ratings.dat'), sep='::', engine='python', names=r_names, encoding='latin-1')
    users = pd.read_csv(os.path.join(MOVIELENS_PATH, 'users.dat'), sep='::', engine='python', names=u_names, encoding='latin-1')
    movies = pd.read_csv(os.path.join(MOVIELENS_PATH, 'movies.dat'), sep='::', engine='python', names=m_names, encoding='latin-1')
    
    # Now, select only the columns we need to keep the dataframe clean
    ratings = ratings[['UserID', 'MovieID', 'Rating']]
    users = users[['UserID', 'Gender', 'Age', 'Occupation']]
    # movies dataframe is already in the desired format
    # --- FIX ENDS HERE ---

    print("All .dat files loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Could not find data files. Make sure MOVIELENS_PATH is set correctly. Missing file: {e.filename}")
    exit()

df = pd.merge(pd.merge(ratings, users, on='UserID'), movies, on='MovieID')
print("Data merged correctly.")
print(f"Shape after initial merge: {df.shape}")


# --- 3. FETCHING POSTER PATHS ---
intermediate_file = 'movies_with_posters.csv'
if os.path.exists(intermediate_file):
    print("\nStep 2: Loading existing movie data with posters...")
    movies_with_posters = pd.read_csv(intermediate_file)
else:
    print("\nStep 2: Fetching poster paths from TMDb (this will take several minutes)...")
    if API_KEY == 'YOUR_API_KEY_HERE':
        print("WARNING: API_KEY is not set. Cannot fetch posters.")
        exit()
        
    def get_poster_path(movie_title, api_key):
        match = re.match(r'^(.*) \((\d{4})\)$', movie_title)
        if not match: return None
        title, year = match.groups()
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {'api_key': api_key, 'query': title, 'primary_release_year': year}
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data['results']: return data['results'][0].get('poster_path')
        except requests.exceptions.RequestException: return None
        return None

    unique_movies = movies.drop_duplicates(subset=['MovieID']).copy()
    poster_paths = []
    for index, row in unique_movies.iterrows():
        path = get_poster_path(row['Title'], API_KEY)
        poster_paths.append(path)
        print(f"Progress: {len(poster_paths)}/{len(unique_movies)}", end='\r')
        time.sleep(0.1)
    
    unique_movies['Poster_Path'] = poster_paths
    movies_with_posters = unique_movies[['MovieID', 'Poster_Path']]
    movies_with_posters.to_csv(intermediate_file, index=False)
    print("\nFinished fetching and saved poster paths to movies_with_posters.csv")

df = pd.merge(df, movies_with_posters, on='MovieID')
df.dropna(subset=['Poster_Path'], inplace=True)
print(f"Shape after merging posters: {df.shape}")


# --- 4. FEATURE ENGINEERING & PREPROCESSING ---
print("\nStep 3: Starting Feature Engineering and Preprocessing...")

df['Year'] = df['Title'].str.extract(r'\((\d{4})\)$', expand=False)
df.dropna(subset=['Year'], inplace=True)
df['Year'] = pd.to_numeric(df['Year'])

df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

user_map = {id: i for i, id in enumerate(df['UserID'].unique())}
movie_map = {id: i for i, id in enumerate(df['MovieID'].unique())}
occupation_map = {id: i for i, id in enumerate(df['Occupation'].unique())}

df['UserIndex'] = df['UserID'].map(user_map)
df['MovieIndex'] = df['MovieID'].map(movie_map)
df['OccupationIndex'] = df['Occupation'].map(occupation_map)

genres_df = df['Genres'].str.get_dummies(sep='|')
df = pd.concat([df, genres_df], axis=1)

scaler = MinMaxScaler()
df[['Age_Scaled', 'Year_Scaled']] = scaler.fit_transform(df[['Age', 'Year']])

# --- 5. FINALIZE AND SAVE ---
print("\nStep 4: Finalizing and saving the dataset...")

final_cols = [
    'UserID', 'MovieID', 'UserIndex', 'MovieIndex', 'Rating', 'Gender', 
    'Age_Scaled', 'OccupationIndex', 'Year_Scaled', 'Poster_Path'
] + genres_df.columns.tolist()

final_df = df[final_cols]

final_df.to_csv(FINAL_OUTPUT_FILE, index=False)

print(f"\nSUCCESS! Fully preprocessed data saved to '{FINAL_OUTPUT_FILE}'")
print(f"Final shape of the dataset is: {final_df.shape}")
print("Please check the first few rows:")
print(final_df.head())
