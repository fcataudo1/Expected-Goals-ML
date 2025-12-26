# src/feature_engineering.py
import numpy as np
import pandas as pd

GOAL_X = 120
GOAL_Y = 40
POST_LEFT_Y = 36
POST_RIGHT_Y = 44

def compute_distance(x, y):
    return np.sqrt((GOAL_X - x)**2 + (GOAL_Y - y)**2)

def compute_angle(x, y):
    left = np.array([GOAL_X - x, POST_LEFT_Y - y], dtype=float)
    right = np.array([GOAL_X - x, POST_RIGHT_Y - y], dtype=float)

    dot = np.dot(left, right)
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    if norm == 0:
        return 0.0

    cosang = dot / norm
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.arccos(cosang))

def build_features(df):
    if df.empty:
        raise ValueError("DataFrame vuoto")

    df = df.dropna(subset=["x", "y"])

    df["distance"] = df.apply(lambda r: compute_distance(r.x, r.y), axis=1)
    df["angle"] = df.apply(lambda r: compute_angle(r.x, r.y), axis=1)

    # NON eliminare statsbomb_xg (serve per confronto a posteriori)
    df = df.drop(columns=["x", "y"])

    return df

