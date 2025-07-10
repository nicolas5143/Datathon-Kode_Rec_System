# Filename: 2_train_model_pytorch.py
# Description: This script can train either the full multimodal model or an
# ablated (no-vision) version to evaluate the impact of the poster images.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from PIL import Image
import requests
from io import BytesIO
import os
import sys # for getting the parameter file to run this file automatically for using vision tower and not
import timm
from tqdm import tqdm

tqdm.pandas(desc="Caching Posters")

# --- CONFIGURATION ---
# --- ABLATION STUDY CONTROL ---
# Set to True to train the full model with the vision tower.
# Set to False to train the simpler model WITHOUT the vision tower.
USE_VISION_TOWER = None
if sys.argv[1]=='vision':
    USE_VISION_TOWER = True
elif sys.argv[1]=='no_vision':
    USE_VISION_TOWER = False

DATA_FILE = 'preprocessed_movie_data.csv'
MODEL_OUTPUT_FILE = f'movie_recommender_model_{"vision" if USE_VISION_TOWER else "no_vision"}.pth'
IMAGE_DIR = 'posters/'
IMAGE_WIDTH = 75
IMAGE_HEIGHT = 110
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
EPOCHS = 5
SAMPLES_PER_EPOCH = 50000 

# Checking where the model is saved
if USE_VISION_TOWER:
    print('Doing training on the model using the vision tower')
else:
    print('Doing training on the model without using the vision tower')

print('Saved the model in:', MODEL_OUTPUT_FILE)

# --- DEFINITIONS ---
def get_image_path(poster_path_suffix):
    if pd.isna(poster_path_suffix): return None
    filename = os.path.basename(poster_path_suffix)
    local_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(local_path):
        image_url = f"https://image.tmdb.org/t/p/w500{poster_path_suffix}"
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        except Exception: return None
    return local_path

class MovieDataset(Dataset):
    def __init__(self, dataframe, user_cols, movie_cols, genre_cols, use_vision=True):
        self.df = dataframe
        self.user_cols, self.movie_cols, self.genre_cols = user_cols, movie_cols, genre_cols
        self.use_vision = use_vision
    
    def __len__(self): return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {
            'user_features': torch.tensor(row[self.user_cols].values.astype(np.float32)),
            'movie_features': torch.tensor(row[self.movie_cols].values.astype(np.float32)),
            'genre_features': torch.tensor(row[self.genre_cols].values.astype(np.float32)),
            'rating': torch.tensor(row['Rating'], dtype=torch.float32)
        }
        if self.use_vision:
            try:
                image = Image.open(row['Local_Poster_Path']).convert('RGB')
                image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                image = np.array(image, dtype=np.float32) / 255.0
                item['poster_image'] = torch.tensor(image).permute(2, 0, 1)
            except Exception:
                item['poster_image'] = torch.zeros((3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32)
        return item

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
            self.vision_backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
            for param in self.vision_backbone.parameters(): param.requires_grad = False
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

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    print(f"--- Starting Experiment: USE_VISION_TOWER = {USE_VISION_TOWER} ---")
    print(f"Using device: {DEVICE}")

    df = pd.read_csv(DATA_FILE)
    if USE_VISION_TOWER:
        if not os.path.exists(IMAGE_DIR): os.makedirs(IMAGE_DIR)
        unique_movies_df = df[['MovieID', 'Poster_Path']].drop_duplicates().copy()
        unique_movies_df['Local_Poster_Path'] = unique_movies_df['Poster_Path'].progress_apply(get_image_path)
        df = pd.merge(df, unique_movies_df[['MovieID', 'Local_Poster_Path']], on='MovieID', how='left')
        df.dropna(subset=['Local_Poster_Path'], inplace=True)
    
    user_cols = ['UserIndex', 'Gender', 'Age_Scaled', 'OccupationIndex']
    movie_cols = ['MovieIndex', 'Year_Scaled']
    genre_cols = [col for col in df.columns if col.startswith(('Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'))]
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = MovieDataset(train_df, user_cols, movie_cols, genre_cols, use_vision=USE_VISION_TOWER)
    val_dataset = MovieDataset(val_df.sample(20000), user_cols, movie_cols, genre_cols, use_vision=USE_VISION_TOWER)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    num_users, num_movies, num_occupations, num_genres = df['UserIndex'].max() + 1, df['MovieIndex'].max() + 1, df['OccupationIndex'].max() + 1, len(genre_cols)
    model = MultimodalRecommender(num_users, num_movies, num_occupations, num_genres, use_vision=USE_VISION_TOWER).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        if SAMPLES_PER_EPOCH and SAMPLES_PER_EPOCH < len(train_dataset):
            sampler = SubsetRandomSampler(np.random.choice(len(train_dataset), SAMPLES_PER_EPOCH, replace=False))
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for batch in train_progress_bar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            ratings = batch.pop('rating')
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            train_progress_bar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
        with torch.no_grad():
            for batch in val_progress_bar:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                ratings = batch.pop('rating')
                outputs = model(batch)
                loss = criterion(outputs, ratings)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary: Validation RMSE: {np.sqrt(avg_val_loss):.4f}\n")

    torch.save(model.state_dict(), MODEL_OUTPUT_FILE)
    print(f"Model saved to '{MODEL_OUTPUT_FILE}'")
