# Filename: 3_app.py
# Description: A simple Flask web application to load the trained PyTorch model,
# generate movie recommendations, and display evaluation metrics.
# UPDATE: Added Precision@10 and Recall@10 evaluation metrics.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO
import os
import timm
from flask import Flask, render_template_string, request
import random
from sklearn.model_selection import train_test_split

# --- 1. SETUP AND CONFIGURATION ---
DATA_FILE = 'preprocessed_movie_data.csv'
MODEL_FILE = 'movie_recommender_model.pth'
MOVIES_DAT_FILE = 'C:\\Users\\User\\OneDrive\\Documents\\datathon\\recom_system_making\\data\\raw_data\\ml-1m\\movies.dat' 
IMAGE_DIR = 'posters/'
IMAGE_WIDTH = 75 # adjusted to the image width that inputted to the training
IMAGE_HEIGHT = 110 # adjusted to the image height that inputted to the training
DEVICE = 'cpu'

# --- 2. LOAD ALL NECESSARY DATA AND MODELS ---
print("--- Loading models and data. This may take a moment... ---")

# --- Define the Model Class (must be identical to the one used for training) ---
class MultimodalRecommender(nn.Module):
    def __init__(self, num_users, num_movies, num_occupations, num_genres):
        super().__init__()
        user_emb_size, occ_emb_size, movie_emb_size = 16, 8, 16
        user_tower_size, movie_tower_size, genre_tower_size, vision_tower_size = 32, 32, 16, 32
        self.user_embedding = nn.Embedding(num_users, user_emb_size)
        self.occupation_embedding = nn.Embedding(num_occupations, occ_emb_size)
        self.user_tower = nn.Sequential(nn.Linear(user_emb_size + 1 + 1 + occ_emb_size, user_tower_size), nn.ReLU())
        self.movie_embedding = nn.Embedding(num_movies, movie_emb_size)
        self.movie_tower = nn.Sequential(nn.Linear(movie_emb_size + 1, movie_tower_size), nn.ReLU())
        self.genre_tower = nn.Sequential(nn.Linear(num_genres, genre_tower_size), nn.ReLU())
        self.vision_backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.vision_tower = nn.Sequential(nn.Linear(self.vision_backbone.num_features, vision_tower_size), nn.ReLU())
        fusion_input_size = user_tower_size + movie_tower_size + genre_tower_size + vision_tower_size
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def forward(self, user_features, movie_features, genre_features, poster_image):
        user_idx, gender, age, occ_idx = user_features[:, 0].long(), user_features[:, 1].unsqueeze(1), user_features[:, 2].unsqueeze(1), user_features[:, 3].long()
        user_emb, occ_emb = self.user_embedding(user_idx), self.occupation_embedding(occ_idx)
        user_output = self.user_tower(torch.cat([user_emb, gender, age, occ_emb], dim=1))
        movie_idx, year = movie_features[:, 0].long(), movie_features[:, 1].unsqueeze(1)
        movie_emb = self.movie_embedding(movie_idx)
        movie_output = self.movie_tower(torch.cat([movie_emb, year], dim=1))
        genre_output = self.genre_tower(genre_features)
        vision_output = self.vision_tower(self.vision_backbone(poster_image))
        fused = torch.cat([user_output, movie_output, genre_output, vision_output], dim=1)
        return self.fusion_layer(fused).squeeze(1)

# Load the preprocessed data
df = pd.read_csv(DATA_FILE)

# Load original movie titles and genres
movies_df = pd.read_csv(MOVIES_DAT_FILE, sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
df = pd.merge(df, movies_df[['MovieID', 'Title', 'Genres']], on='MovieID', how='left')

# --- NEW: Recreate train/validation split for evaluation ---
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Get model parameters from the full dataframe
num_users = df['UserIndex'].max() + 1
num_movies = df['MovieIndex'].max() + 1
num_occupations = df['OccupationIndex'].max() + 1
genre_cols = [col for col in df.columns if col.startswith(('Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'))]
num_genres = len(genre_cols)

# Instantiate the model and load the trained weights
model = MultimodalRecommender(num_users, num_movies, num_occupations, num_genres)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("--- Models and data loaded successfully. Starting Flask app. ---")

# --- 3. FLASK WEB APPLICATION ---
app = Flask(__name__)

def prepare_input_data(user_data, movie_data):
    user_features = torch.tensor([user_data[['UserIndex', 'Gender', 'Age_Scaled', 'OccupationIndex']].values], dtype=torch.float32).to(DEVICE)
    movie_features = torch.tensor([movie_data[['MovieIndex', 'Year_Scaled']].values], dtype=torch.float32).to(DEVICE)
    genre_features = torch.tensor([movie_data[genre_cols].values.astype(np.float32)], dtype=torch.float32).to(DEVICE)
    image_path = os.path.join(IMAGE_DIR, os.path.basename(movie_data['Poster_Path']))
    image = Image.open(image_path).convert('RGB')
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = np.array(image, dtype=np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return user_features, movie_features, genre_features, image

@app.route('/', methods=['GET', 'POST'])
def home():
    user_ids = sorted(df['UserID'].unique())
    selected_user_id = None
    recommendations = []
    watched_movies = []
    message = ""
    evaluation_results = {} # Dictionary to hold evaluation metrics

    if request.method == 'POST':
        if 'random_user' in request.form:
            selected_user_id = random.choice(user_ids)
        elif request.form.get('user_id'):
            selected_user_id = int(request.form['user_id'])
        
        if selected_user_id:
            # --- EVALUATION LOGIC ---
            # Use the validation set for the selected user
            user_val_data = val_df[val_df['UserID'] == selected_user_id]
            if not user_val_data.empty:
                # Ground truth: movies the user actually liked in the validation set
                relevant_items = set(user_val_data[user_val_data['Rating'] >= 4.0]['MovieID'])
                
                # Get model predictions for all movies in this user's validation set
                val_predictions = []
                with torch.no_grad():
                    for _, movie_data in user_val_data.iterrows():
                        input_data = prepare_input_data(movie_data, movie_data)
                        user_feat, movie_feat, genre_feat, poster_img = input_data
                        pred_rating = model(user_feat, movie_feat, genre_feat, poster_img).item()
                        val_predictions.append({'MovieID': movie_data['MovieID'], 'PredictedRating': pred_rating})
                
                # Top-K recommended items from the validation set
                val_predictions.sort(key=lambda x: x['PredictedRating'], reverse=True)
                top_k_recs = set([pred['MovieID'] for pred in val_predictions[:10]])
                
                # Calculate metrics
                hits = len(top_k_recs.intersection(relevant_items))
                precision_at_10 = hits / 10.0
                recall_at_10 = hits / len(relevant_items) if relevant_items else 0.0
                evaluation_results = {'precision': precision_at_10, 'recall': recall_at_10}

            # --- DEMONSTRATION LOGIC (existing code) ---
            user_history_df = train_df[train_df['UserID'] == selected_user_id].sort_values('Rating', ascending=False).head(10)
            for _, movie in user_history_df.iterrows():
                watched_movies.append({'Title': movie['Title'], 'Rating': movie['Rating'], 'PosterURL': f"https://image.tmdb.org/t/p/w500{movie['Poster_Path']}", 'Genres': movie['Genres'].replace('|', ', ')})

            seen_movie_ids = df[df['UserID'] == selected_user_id]['MovieID'].unique()
            unseen_movies_df = df[~df['MovieID'].isin(seen_movie_ids)].drop_duplicates(subset=['MovieID']).copy()
            unseen_movies_df['local_path'] = unseen_movies_df['Poster_Path'].apply(lambda x: os.path.join(IMAGE_DIR, os.path.basename(x)))
            unseen_movies_df = unseen_movies_df[unseen_movies_df['local_path'].apply(os.path.exists)]

            predictions = []
            if not unseen_movies_df.empty:
                sample_size = min(len(unseen_movies_df), 200)
                with torch.no_grad():
                    progress_bar = tqdm(unseen_movies_df.sample(n=sample_size, random_state=42).iterrows(), total=sample_size, desc=f"Generating for User {selected_user_id}")
                    for _, movie_data in progress_bar:
                        input_data = prepare_input_data(movie_data, movie_data)
                        user_feat, movie_feat, genre_feat, poster_img = input_data
                        pred_rating = model(user_feat, movie_feat, genre_feat, poster_img).item()
                        predictions.append({'Title': movie_data['Title'], 'PredictedRating': pred_rating, 'PosterURL': f"https://image.tmdb.org/t/p/w500{movie_data['Poster_Path']}", 'Genres': movie_data['Genres'].replace('|', ', ')})
                recommendations = sorted(predictions, key=lambda x: x['PredictedRating'], reverse=True)[:10]
            
            if not recommendations: message = "Could not generate new recommendations for this user."

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Movie Recommender</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f0f2f5; color: #1c1e21; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ text-align: center; color: #007bff; }}
            h2 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 40px;}}
            form {{ display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 30px; }}
            select, button {{ font-size: 16px; padding: 10px; border-radius: 6px; border: 1px solid #ddd; }}
            button {{ background-color: #007bff; color: white; cursor: pointer; border: none; }}
            button.secondary {{ background-color: #6c757d; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }}
            .movie-card {{ background: #f9f9f9; border: 1px solid #eee; border-radius: 8px; padding: 10px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05); display: flex; flex-direction: column; }}
            .movie-card img {{ max-width: 100%; border-radius: 6px; }}
            .movie-card h3 {{ font-size: 16px; margin: 10px 0 5px 0; flex-grow: 1; }}
            .movie-card p {{ font-size: 14px; color: #606770; margin: 0; }}
            .movie-card p.genres {{ font-size: 12px; color: #8a8d91; margin-top: 5px; font-style: italic; }}
            .message {{ text-align: center; color: #606770; font-size: 18px; }}
            .evaluation-box {{ background-color: #e7f3ff; border: 1px solid #cce5ff; border-radius: 8px; padding: 15px; margin: 20px 0; text-align: center; }}
            .evaluation-box h3 {{ margin-top: 0; color: #004085; }}
            .evaluation-box p {{ font-size: 18px; color: #004085; margin: 5px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Personalized Movie Recommender</h1>
            <form method="post">
                <label for="user_id">Select a User ID:</label>
                <select name="user_id" id="user_id">
                    <option value="">--Select User--</option>
                    {''.join([f'<option value="{uid}" {"selected" if uid == selected_user_id else ""}>{uid}</option>' for uid in user_ids])}
                </select>
                <button type="submit">Get Recommendations</button>
                <button type="submit" name="random_user" value="true" class="secondary">Get Random User</button>
            </form>

            {f'''<div class="evaluation-box">
                    <h3>Evaluation Metrics for User {selected_user_id}</h3>
                    <p><b>Precision@10:</b> {evaluation_results.get("precision", 0.0):.2f}</p>
                    <p><b>Recall@10:</b> {evaluation_results.get("recall", 0.0):.2f}</p>
                </div>''' if evaluation_results else ''}

            {'<h2>User ' + str(selected_user_id) + '\'s Highest Rated Movies (from Training Set)</h2><div class="grid">' + ''.join([f'''
                <div class="movie-card">
                    <img src="{movie['PosterURL']}" alt="{movie['Title']}">
                    <h3>{movie['Title']}</h3>
                    <p>Your Rating: {movie['Rating']}</p>
                    <p class="genres">{movie['Genres']}</p>
                </div>
            ''' for movie in watched_movies]) + '</div>' if watched_movies else ''}

            {'<h2>Top 10 Recommendations for User ' + str(selected_user_id) + '</h2>' if recommendations else ''}
            {f'<p class="message">{message}</p>' if message else ''}
            <div class="grid">
                {''.join([f'''
                    <div class="movie-card">
                        <img src="{rec['PosterURL']}" alt="{rec['Title']}">
                        <h3>{rec['Title']}</h3>
                        <p>Predicted Rating: {rec['PredictedRating']:.2f}</p>
                        <p class="genres">{rec['Genres']}</p>
                    </div>
                ''' for rec in recommendations])}
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template)

# --- Run the App ---
if __name__ == '__main__':
    from tqdm import tqdm
    app.run(debug=True, port=5001)
