"""
train_save.py – Train NCF and persist all inference artefacts
=============================================================
Run this ONCE locally before deploying.  It produces model_artifacts.pt,
which app.py loads at startup (no dataset needed at runtime).

Usage (from the project root):
    python train_save.py

Artefacts saved in model_artifacts.pt
--------------------------------------
  model_state_dict  : trained weights
  user_to_idx       : {raw_user_id  → embedding index}
  movie_to_idx      : {raw_movie_id → embedding index}
  idx_to_movie      : {embedding index → raw_movie_id}
  movie_meta        : {raw_movie_id → {title, genres}}
  n_users / n_movies: int
  min_rating / max_rating: float (rating de-normalisation scale)
  embedding_dim / mlp_layers / dropout: model hyper-params
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import the shared model class
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import NeuralCollaborativeFiltering

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SEED          = 42
EMBEDDING_DIM = 32
MLP_LAYERS    = [64, 32, 16]
DROPOUT       = 0.2
LR            = 0.001
BATCH_SIZE    = 256
NUM_EPOCHS    = 20

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts.pt")

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")   # keep deterministic; CPU is fine for this dataset

# ── Locate dataset ────────────────────────────────────────────────────────────
for _candidate in ["./dataset", "../dataset", "dataset"]:
    if os.path.isfile(os.path.join(_candidate, "ratings.csv")):
        DATASET_DIR = _candidate
        break
else:
    sys.exit(
        "[ERROR] Dataset not found.  "
        "Expected ratings.csv / movies.csv inside a 'dataset/' folder."
    )

print(f"Dataset dir : {DATASET_DIR}")
ratings = pd.read_csv(os.path.join(DATASET_DIR, "ratings.csv"))
movies  = pd.read_csv(os.path.join(DATASET_DIR, "movies.csv"))
print(f"  Ratings   : {len(ratings):,}")
print(f"  Movies    : {len(movies):,}")

# ── Encode IDs ────────────────────────────────────────────────────────────────
user_enc  = LabelEncoder()
movie_enc = LabelEncoder()
ratings["user_idx"]  = user_enc.fit_transform(ratings["userId"])
ratings["movie_idx"] = movie_enc.fit_transform(ratings["movieId"])

n_users  = int(ratings["user_idx"].nunique())
n_movies = int(ratings["movie_idx"].nunique())
min_r    = float(ratings["rating"].min())
max_r    = float(ratings["rating"].max())

ratings["rating_norm"] = (ratings["rating"] - min_r) / (max_r - min_r)
print(f"  Users     : {n_users}  |  Movies  : {n_movies}")
print(f"  Rating range: [{min_r}, {max_r}]")

# ── Dataset / DataLoaders ─────────────────────────────────────────────────────
class _RatingDS(Dataset):
    def __init__(self, df):
        self.u = torch.LongTensor(df["user_idx"].values)
        self.m = torch.LongTensor(df["movie_idx"].values)
        self.r = torch.FloatTensor(df["rating_norm"].values)

    def __len__(self):            return len(self.u)
    def __getitem__(self, idx):   return self.u[idx], self.m[idx], self.r[idx]


tr_df, tmp = train_test_split(ratings, test_size=0.2, random_state=SEED)
vl_df, _   = train_test_split(tmp,     test_size=0.5, random_state=SEED)

train_loader = DataLoader(_RatingDS(tr_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(_RatingDS(vl_df), batch_size=BATCH_SIZE, shuffle=False)

# ── Build and train model ─────────────────────────────────────────────────────
model     = NeuralCollaborativeFiltering(n_users, n_movies, EMBEDDING_DIM, MLP_LAYERS, DROPOUT)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

best_val_loss = float("inf")
best_state    = None

print(f"\nTraining for {NUM_EPOCHS} epochs …")
print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}")
print("-" * 30)

for ep in range(1, NUM_EPOCHS + 1):
    # --- train ---
    model.train()
    tr_loss = 0.0
    for u, m, r in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(u, m), r)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
    tr_loss /= len(train_loader)

    # --- validate ---
    model.eval()
    vl_loss = 0.0
    with torch.no_grad():
        for u, m, r in val_loader:
            vl_loss += criterion(model(u, m), r).item()
    vl_loss /= len(val_loader)

    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    scheduler.step()

    if ep % 5 == 0 or ep == 1:
        print(f"{ep:>6}  {tr_loss:>10.4f}  {vl_loss:>10.4f}")

model.load_state_dict(best_state)
print(f"\nBest validation MSE : {best_val_loss:.4f}")

# ── Build lookup tables ───────────────────────────────────────────────────────
user_to_idx  = {int(uid): int(i) for i, uid in enumerate(user_enc.classes_)}
movie_to_idx = {int(mid): int(i) for i, mid in enumerate(movie_enc.classes_)}
idx_to_movie = {v: k for k, v in movie_to_idx.items()}
movie_meta   = {
    int(row["movieId"]): {"title": row["title"], "genres": row["genres"]}
    for _, row in movies.iterrows()
}

# ── Save artefacts ────────────────────────────────────────────────────────────
torch.save(
    {
        "model_state_dict": best_state,
        "user_to_idx":      user_to_idx,
        "movie_to_idx":     movie_to_idx,
        "idx_to_movie":     idx_to_movie,
        "movie_meta":       movie_meta,
        "n_users":          n_users,
        "n_movies":         n_movies,
        "min_rating":       min_r,
        "max_rating":       max_r,
        "embedding_dim":    EMBEDDING_DIM,
        "mlp_layers":       MLP_LAYERS,
        "dropout":          DROPOUT,
    },
    OUTPUT_PATH,
)

print(f"\nSaved → {OUTPUT_PATH}")
print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
print(f"  Users      : {n_users}  |  Movies : {n_movies}")
print("\nNext steps:")
print("  1. git add model_artifacts.pt && git commit -m 'add model artifacts'")
print("  2. git push")
print("  3. Deploy the repo on Render.com (see Module5_Documentation.md)")
