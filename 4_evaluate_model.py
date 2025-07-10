# Filename: 4_evaluate_model.py
# Description: This script loads a trained model (either with or without the vision tower)
# and computes the overall average Precision@10, Recall@10, and F1-Score@10 across all users.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import sys

# --- 1. SETUP AND CONFIGURATION ---

# --- ABLATION STUDY CONTROL ---
# Set to True to evaluate the full model with the vision tower.
# Set to False to evaluate the simpler model WITHOUT the vision tower.
USE_VISION_TOWER = None
EVALUATION_FILE = ''
if sys.argv[1]=='vision':
    USE_VISION_TOWER = True
    EVALUATION_FILE = 'evaluation_model_vision.txt'
elif sys.argv[1]=='no_vision':
    USE_VISION_TOWER = False
    EVALUATION_FILE = 'evaluation_model_no_vision.txt'

DATA_FILE = 'preprocessed_movie_data.csv'
# MODEL_FILE = f'movie_recommender_model_{"vision" if USE_VISION_TOWER else "no_vision"}.pth'
MODEL_FILE = 'movie_recommender_model.pth'
IMAGE_DIR = 'posters/'
IMAGE_WIDTH = 75
IMAGE_HEIGHT = 110
DEVICE = 'cpu'
K = 10 # The "K" in Precision@K and Recall@K
RATING_THRESHOLD = 4.0 # What we consider a "good" or "relevant" rating

print('\nSaving model in:', MODEL_FILE)
print('Saving evaluation in:', EVALUATION_FILE, '\n')

# --- 2. LOAD MODELS AND DATA ---
print(f"--- Loading model: {MODEL_FILE} ---")

# --- Define the Model Class (must be identical to the one used for training) ---
class MultimodalRecommender(nn.Module):
    def __init__(self, num_users, num_movies, num_occupations, num_genres, use_vision=True):
        super().__init__()
        self.use_vision = use_vision
        user_emb_size, occ_emb_size, movie_emb_size = 16, 8, 16
        user_tower_size, movie_tower_size, genre_tower_size, vision_tower_size = 32, 32, 16, 32
        
        self.user_embedding = nn.Embedding(num_users, user_emb_size)
        self.occupation_embedding = nn.Embedding(num_occupations, occ_emb_size)
        self.user_tower = nn.Sequential(nn.Linear(user_emb_size + 1 + 1 + occ_emb_size, user_tower_size), nn.ReLU())
        
        self.movie_embedding = nn.Embedding(num_movies, movie_emb_size)
        self.movie_tower = nn.Sequential(nn.Linear(movie_emb_size + 1, movie_tower_size), nn.ReLU())
        
        self.genre_tower = nn.Sequential(nn.Linear(num_genres, genre_tower_size), nn.ReLU())
        
        fusion_input_size = user_tower_size + movie_tower_size + genre_tower_size
        
        if self.use_vision:
            self.vision_backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
            self.vision_tower = nn.Sequential(nn.Linear(self.vision_backbone.num_features, vision_tower_size), nn.ReLU())
            fusion_input_size += vision_tower_size
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def forward(self, batch):
        user_feat, movie_feat, genre_feat = batch['user_features'], batch['movie_features'], batch['genre_features']
        
        user_idx, gender, age, occ_idx = user_feat[:, 0].long(), user_feat[:, 1].unsqueeze(1), user_feat[:, 2].unsqueeze(1), user_feat[:, 3].long()
        user_emb, occ_emb = self.user_embedding(user_idx), self.occupation_embedding(occ_idx)
        user_output = self.user_tower(torch.cat([user_emb, gender, age, occ_emb], dim=1))
        
        movie_idx, year = movie_feat[:, 0].long(), movie_feat[:, 1].unsqueeze(1)
        movie_emb = self.movie_embedding(movie_idx)
        movie_output = self.movie_tower(torch.cat([movie_emb, year], dim=1))
        
        genre_output = self.genre_tower(genre_feat)
        
        feature_towers = [user_output, movie_output, genre_output]
        
        if self.use_vision:
            poster_img = batch['poster_image']
            vision_output = self.vision_tower(self.vision_backbone(poster_img))
            feature_towers.append(vision_output)
        
        fused = torch.cat(feature_towers, dim=1)
        return self.fusion_layer(fused).squeeze(1)

# Load the preprocessed data
df = pd.read_csv(DATA_FILE)

# Recreate the exact same train/validation split as used during training
_, val_df = train_test_split(df, test_size=0.2, random_state=42)
val_df = val_df.sample(40000)

# Get model parameters
num_users = df['UserIndex'].max() + 1
num_movies = df['MovieIndex'].max() + 1
num_occupations = df['OccupationIndex'].max() + 1
genre_cols = [col for col in df.columns if col.startswith(('Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'))]
num_genres = len(genre_cols)

# Instantiate the model and load the trained weights
model = MultimodalRecommender(num_users, num_movies, num_occupations, num_genres, use_vision=USE_VISION_TOWER)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("--- Models and data loaded. Starting evaluation... ---")

# --- 3. EVALUATION LOGIC ---

def prepare_batch(user_data, movie_data):
    """Helper function to format a single row into a batch dictionary for the model."""
    batch = {
        'user_features': torch.tensor([user_data[['UserIndex', 'Gender', 'Age_Scaled', 'OccupationIndex']].values], dtype=torch.float32).to(DEVICE),
        'movie_features': torch.tensor([movie_data[['MovieIndex', 'Year_Scaled']].values], dtype=torch.float32).to(DEVICE),
        'genre_features': torch.tensor([movie_data[genre_cols].values.astype(np.float32)], dtype=torch.float32).to(DEVICE)
    }
    
    if USE_VISION_TOWER:
        image_path = os.path.join(IMAGE_DIR, os.path.basename(movie_data['Poster_Path']))
        if not os.path.exists(image_path): return None
        
        image = Image.open(image_path).convert('RGB')
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        image = np.array(image, dtype=np.float32) / 255.0
        batch['poster_image'] = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
    return batch

all_precisions, all_recalls, all_f1_scores = [], [], []
all_true_ratings, all_pred_ratings = [], [] # For RMSE and MAE

user_groups = val_df.groupby('UserID')
progress_bar = tqdm(user_groups, desc="Evaluating All Users")

for user_id, group in progress_bar:
    relevant_items = set(group[group['Rating'] >= RATING_THRESHOLD]['MovieID'])
    
    val_predictions = []
    with torch.no_grad():
        for _, movie_data in group.iterrows():
            batch = prepare_batch(movie_data, movie_data)
            if batch:
                pred_rating = model(batch).item()
                val_predictions.append({'MovieID': movie_data['MovieID'], 'PredictedRating': pred_rating})
                
                # Store true and predicted ratings for overall metrics
                all_true_ratings.append(movie_data['Rating'])
                all_pred_ratings.append(pred_rating)

    if not relevant_items:
        continue # Skip ranking metrics if user has no "liked" items
        
    val_predictions.sort(key=lambda x: x['PredictedRating'], reverse=True)
    top_k_recs = set([pred['MovieID'] for pred in val_predictions[:K]])
    
    hits = len(top_k_recs.intersection(relevant_items))
    precision = hits / K
    recall = hits / len(relevant_items)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1_scores.append(f1)

# --- 4. DISPLAY FINAL RESULTS ---
print("\n--- Overall Model Evaluation Results ---")
print(f"Model: {'Multimodal (with Vision)' if USE_VISION_TOWER else 'Standard (No Vision)'}")

# Calculate and print rating prediction metrics
if all_true_ratings:
    overall_rmse = np.sqrt(mean_squared_error(all_true_ratings, all_pred_ratings))
    overall_mae = mean_absolute_error(all_true_ratings, all_pred_ratings)

    display_eval = f'Rating Prediction Accuracy:\nOverall Validation RMSE: {overall_rmse:.4f}\nOverall Validation MAE:  {overall_mae:.4f}'

    with open(EVALUATION_FILE, 'a') as file:
        file.write(display_eval)

    print(display_eval)
else:
    print("\nCould not calculate rating accuracy metrics.")

# Calculate and print ranking metrics
if all_precisions:
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1_scores)

    display_eval = f'\n\nTop-{K} Ranking Quality:\nAverage Precision@{K}: {avg_precision:.4f}\nAverage Recall@{K}:    {avg_recall:.4f}\nAverage F1-Score@{K}:  {avg_f1:.4f}'

    with open(EVALUATION_FILE, 'a') as file:
        file.write(display_eval)

    print(display_eval)
else:
    print("\nCould not calculate ranking quality metrics.")

