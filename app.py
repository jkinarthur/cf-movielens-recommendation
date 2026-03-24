"""
app.py – NCF Movie Recommendation Microservice
===============================================
FastAPI application that loads the pre-trained NCF model artefacts and
exposes a REST API for rating prediction and personalised recommendations.

Requires model_artifacts.pt (produced by train_save.py).

Start locally:
    uvicorn app:app --reload --port 8000

Interactive docs available at:
    http://localhost:8000/docs
"""

import os
import sys
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

# Ensure model.py is importable whether the working dir is the project root
# or an alternative location.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import NeuralCollaborativeFiltering


# ── Load artefacts at startup ─────────────────────────────────────────────────
_ARTIFACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts.pt")

if not os.path.exists(_ARTIFACT_PATH):
    raise RuntimeError(
        "model_artifacts.pt not found. "
        "Run `python train_save.py` first, then commit the file before deploying."
    )

# weights_only=False is required because we saved non-tensor Python objects
# (dicts, lists, strings).  The file is our own trusted artefact.
try:
    _art = torch.load(_ARTIFACT_PATH, map_location="cpu", weights_only=False)
except TypeError:
    # Older PyTorch versions (<1.13) do not have the weights_only parameter.
    _art = torch.load(_ARTIFACT_PATH, map_location="cpu")

# Rebuild model
_model = NeuralCollaborativeFiltering(
    n_users      = _art["n_users"],
    n_movies     = _art["n_movies"],
    embedding_dim= _art["embedding_dim"],
    mlp_layers   = _art["mlp_layers"],
    dropout      = _art["dropout"],
)
_model.load_state_dict(_art["model_state_dict"])
_model.eval()

# Lookup tables
_user_to_idx   = _art["user_to_idx"]    # raw user_id  → embedding index
_movie_to_idx  = _art["movie_to_idx"]   # raw movie_id → embedding index
_idx_to_movie  = _art["idx_to_movie"]   # embedding index → raw movie_id
_movie_meta    = _art["movie_meta"]     # raw movie_id → {title, genres}
_MIN_R, _MAX_R = _art["min_rating"], _art["max_rating"]
_N_MOVIES      = _art["n_movies"]

_VALID_USERS  = sorted(_user_to_idx.keys())
_VALID_MOVIES = sorted(_movie_to_idx.keys())


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "NCF Movie Recommendation Service",
    description = (
        "Neural Collaborative Filtering (NCF) microservice trained on the "
        "**MovieLens-Small** dataset "
        "(100 K ratings · 610 users · 9 742 movies).  \n\n"
        "Predicts star ratings on a **0.5 – 5.0** scale and returns "
        "personalised ranked recommendations. "
        "See `/docs` for interactive exploration."
    ),
    version     = "1.0.0",
    contact     = {"name": "Module 5 – NCF Microservice"},
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    user_id:  int
    movie_id: int

    model_config = {
        "json_schema_extra": {
            "example": {"user_id": 1, "movie_id": 1}
        }
    }


class PredictResponse(BaseModel):
    user_id:          int
    movie_id:         int
    movie_title:      str
    genres:           str
    predicted_rating: float


class RecommendationItem(BaseModel):
    rank:             int
    movie_id:         int
    title:            str
    genres:           str
    predicted_rating: float


class RecommendResponse(BaseModel):
    user_id:         int
    n:               int
    recommendations: List[RecommendationItem]


# ── Internal helpers ──────────────────────────────────────────────────────────
def _denorm(arr: np.ndarray) -> np.ndarray:
    """Convert normalised [0,1] predictions back to original rating scale."""
    return arr * (_MAX_R - _MIN_R) + _MIN_R


def _clamp(v: float) -> float:
    """Clamp to valid rating range and round to 3 decimal places."""
    return round(float(max(_MIN_R, min(_MAX_R, v))), 3)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"], summary="Service overview and available endpoints")
def root():
    """
    Welcome page.  Lists all endpoints and points to the interactive docs.
    """
    return {
        "service":         "NCF Movie Recommendation Service",
        "version":         "1.0.0",
        "interactive_docs": "/docs",
        "endpoints": {
            "POST /predict":              "Predict rating for a (user_id, movie_id) pair",
            "GET  /recommend/{user_id}":  "Top-N personalised recommendations for a user",
            "GET  /users":                "List all valid user IDs (paginated)",
            "GET  /movies":               "Browse / search the movie catalogue",
            "GET  /health":               "Service health check",
        },
    }


@app.get("/health", tags=["Info"], summary="Health check")
def health():
    """Returns service status and dataset size info."""
    return {
        "status":   "healthy",
        "model":    "NeuralCollaborativeFiltering",
        "n_users":  len(_VALID_USERS),
        "n_movies": len(_VALID_MOVIES),
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict a user's rating for a specific movie",
)
def predict(req: PredictRequest):
    """
    Given a **user_id** and a **movie_id**, returns the NCF model's predicted
    star rating (1.0 – 5.0 scale).

    - Use `GET /users` to find valid `user_id` values.
    - Use `GET /movies` or `GET /movies?search=<title>` to find `movie_id` values.

    **Example request body:**
    ```json
    {"user_id": 1, "movie_id": 1}
    ```
    """
    if req.user_id not in _user_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"user_id {req.user_id} not in training data.  "
                   f"Call GET /users to see valid IDs.",
        )
    if req.movie_id not in _movie_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"movie_id {req.movie_id} not in training data.  "
                   f"Call GET /movies to search by title.",
        )

    with torch.no_grad():
        raw_pred = _model(
            torch.LongTensor([_user_to_idx[req.user_id]]),
            torch.LongTensor([_movie_to_idx[req.movie_id]]),
        ).item()

    rating = _clamp(_denorm(np.array([raw_pred]))[0])
    meta   = _movie_meta.get(req.movie_id, {})

    return PredictResponse(
        user_id          = req.user_id,
        movie_id         = req.movie_id,
        movie_title      = meta.get("title", "Unknown"),
        genres           = meta.get("genres", "Unknown"),
        predicted_rating = rating,
    )


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendResponse,
    tags=["Recommendation"],
    summary="Top-N personalised movie recommendations for a user",
)
def recommend(
    user_id: int,
    n: int = Query(default=10, ge=1, le=50, description="Number of results (1–50)"),
):
    """
    Scores the entire movie catalogue against **user_id** using the NCF model
    and returns the top **n** movies ordered by predicted rating.

    The optional query parameter `n` controls how many results to return
    (default 10, max 50).

    **Example:** `/recommend/1?n=5`
    """
    if user_id not in _user_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"user_id {user_id} not in training data.  "
                   f"Call GET /users to see valid IDs.",
        )

    u_idx = _user_to_idx[user_id]

    with torch.no_grad():
        preds_norm = _model(
            torch.LongTensor([u_idx] * _N_MOVIES),
            torch.LongTensor(list(range(_N_MOVIES))),
        ).numpy()

    scores  = _denorm(preds_norm)
    top_idx = np.argsort(scores)[-n:][::-1]

    recommendations = []
    for rank, idx in enumerate(top_idx, start=1):
        movie_id = _idx_to_movie[int(idx)]
        meta     = _movie_meta.get(movie_id, {})
        recommendations.append(
            RecommendationItem(
                rank             = rank,
                movie_id         = movie_id,
                title            = meta.get("title",  "Unknown"),
                genres           = meta.get("genres", "Unknown"),
                predicted_rating = _clamp(float(scores[idx])),
            )
        )

    return RecommendResponse(user_id=user_id, n=n, recommendations=recommendations)


@app.get("/users", tags=["Info"], summary="List all valid user IDs")
def list_users(
    page:     int = Query(default=1,   ge=1,                description="Page number"),
    per_page: int = Query(default=100, ge=1, le=500,        description="Results per page"),
):
    """
    Returns a paginated list of user IDs that the model was trained on.
    Pass any of these as `user_id` to `/predict` or `/recommend/{user_id}`.
    """
    start = (page - 1) * per_page
    return {
        "total":    len(_VALID_USERS),
        "page":     page,
        "per_page": per_page,
        "user_ids": _VALID_USERS[start : start + per_page],
    }


@app.get("/movies", tags=["Info"], summary="Browse or search the movie catalogue")
def list_movies(
    search:   Optional[str] = Query(default=None, description="Case-insensitive title substring search"),
    page:     int = Query(default=1,  ge=1),
    per_page: int = Query(default=50, ge=1, le=200),
):
    """
    Returns movies that the model knows about, with their `movie_id`, `title`,
    and `genres`.  Use the optional **search** parameter to filter by title
    (e.g. `?search=toy+story`).

    Pass any `movie_id` from this list to `/predict`.
    """
    items = [
        {"movie_id": mid, "title": m["title"], "genres": m["genres"]}
        for mid, m in _movie_meta.items()
    ]
    if search:
        q     = search.lower()
        items = [it for it in items if q in it["title"].lower()]

    items.sort(key=lambda x: x["movie_id"])
    start = (page - 1) * per_page

    return {
        "total":    len(items),
        "page":     page,
        "per_page": per_page,
        "movies":   items[start : start + per_page],
    }
