"""
Robust WTA data ingestion: 2022-2025, point-level features, rolling 12m stats,
min-match threshold per surface, no look-ahead. Output for model_core and backtest.
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import timedelta

GITHUB_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
MIN_MATCHES_SURFACE = 5
ROLLING_MONTHS = 12


def download_csv(year, data_dir="data"):
    fname = f"wta_matches_{year}.csv"
    path = os.path.join(data_dir, fname)
    url = f"{GITHUB_BASE}/{fname}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    os.makedirs(data_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def load_and_join(years=(2022, 2023, 2024, 2025), data_dir="data"):
    dfs = []
    for year in years:
        path = os.path.join(data_dir, f"wta_matches_{year}.csv")
        if not os.path.exists(path):
            download_csv(year, data_dir)
        dfs.append(pd.read_csv(path, low_memory=False))
    out = pd.concat(dfs, ignore_index=True)
    return out


def clean(df):
    out = df.copy()
    if "score" in out.columns:
        out = out[out["score"].notna()]
    if "round" in out.columns:
        out = out[out["round"].notna()]
    if "minutes" in out.columns:
        out["minutes"] = out["minutes"].fillna(out["minutes"].median())
    for col in out.columns:
        if "rank" in col.lower() and out[col].isna().any():
            out[col] = out[col].fillna(999)
    if "tourney_date" in out.columns:
        out["tourney_date"] = pd.to_datetime(out["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
        out = out[out["tourney_date"].notna()]
    return out


def normalize_surface(s):
    if pd.isna(s):
        return "Hard"
    u = str(s).strip().upper()
    if u == "CLAY": return "Clay"
    if u == "GRASS": return "Grass"
    if u == "CARPET": return "Carpet"
    return "Hard"


def point_level_features(df):
    """Serve Win % and Return Win % per match (0-1 scale)."""
    out = df.copy()
    out["surface_type"] = out["surface"].map(normalize_surface) if "surface" in out.columns else "Hard"

    # w_serve = (1stWon + 2ndWon) / svpt
    if all(c in out.columns for c in ["w_1stWon", "w_2ndWon", "w_svpt"]):
        out["winner_serve_win_pct"] = ((out["w_1stWon"] + out["w_2ndWon"]) / out["w_svpt"]).fillna(0)
    else:
        out["winner_serve_win_pct"] = np.nan
    if all(c in out.columns for c in ["l_1stWon", "l_2ndWon", "l_svpt"]):
        out["loser_serve_win_pct"] = ((out["l_1stWon"] + out["l_2ndWon"]) / out["l_svpt"]).fillna(0)
    else:
        out["loser_serve_win_pct"] = np.nan

    # w_return = points won when opponent served / opponent svpt
    if all(c in out.columns for c in ["l_svpt", "l_1stWon", "l_2ndWon"]):
        out["winner_return_win_pct"] = ((out["l_svpt"] - out["l_1stWon"] - out["l_2ndWon"]) / out["l_svpt"]).fillna(0)
    else:
        out["winner_return_win_pct"] = np.nan
    if all(c in out.columns for c in ["w_svpt", "w_1stWon", "w_2ndWon"]):
        out["loser_return_win_pct"] = ((out["w_svpt"] - out["w_1stWon"] - out["w_2ndWon"]) / out["w_svpt"]).fillna(0)
    else:
        out["loser_return_win_pct"] = np.nan

    return out


def player_id_column(df):
    """Stable player_id: use winner_id/loser_id if present, else normalized name."""
    if "winner_id" in df.columns and "loser_id" in df.columns and df["winner_id"].notna().all() and df["loser_id"].notna().all():
        return
    # fallback: name as id (Sackmann names are stable)
    if "winner_id" not in df.columns:
        df["winner_id"] = df["winner_name"].astype(str).str.strip()
    if "loser_id" not in df.columns:
        df["loser_id"] = df["loser_name"].astype(str).str.strip()


def rolling_12m_stats(history_df, player_id, surface, as_of_date, min_matches, window_months):
    """
    Rolling average of serve_win_pct and return_win_pct for player on surface,
    using only matches with match_date < as_of_date, in the last window_months.
    history_df must have columns: winner_id, loser_id, surface_type, tourney_date,
    winner_serve_win_pct, winner_return_win_pct, loser_serve_win_pct, loser_return_win_pct.
    Returns (serve_win_pct, return_win_pct) or (np.nan, np.nan) if no data.
    """
    cutoff = as_of_date - timedelta(days=window_months * 31)
    prior = history_df[history_df["tourney_date"] < as_of_date]
    prior = prior[prior["tourney_date"] >= cutoff]

    w = prior[prior["winner_id"] == player_id][["surface_type", "winner_serve_win_pct", "winner_return_win_pct"]].rename(
        columns={"winner_serve_win_pct": "serve", "winner_return_win_pct": "return"}
    )
    w["surface_type"] = w["surface_type"].astype(str)
    l = prior[prior["loser_id"] == player_id][["surface_type", "loser_serve_win_pct", "loser_return_win_pct"]].rename(
        columns={"loser_serve_win_pct": "serve", "loser_return_win_pct": "return"}
    )
    l["surface_type"] = l["surface_type"].astype(str)
    both = pd.concat([w, l], ignore_index=True)

    # surface-specific
    on_surface = both[both["surface_type"] == surface]
    if len(on_surface) >= min_matches:
        return on_surface["serve"].mean(), on_surface["return"].mean()

    # global fallback
    if len(both) >= 1:
        return both["serve"].mean(), both["return"].mean()
    return np.nan, np.nan


def attach_rolling_stats(df, min_matches=MIN_MATCHES_SURFACE, window_months=ROLLING_MONTHS):
    """
    For each row, attach winner/loser rolling 12m serve_win_pct and return_win_pct
    using only matches strictly before this match's date (no look-ahead).
    """
    df = df.sort_values("tourney_date").reset_index(drop=True)
    w_serve = []
    w_return = []
    l_serve = []
    l_return = []
    for i in range(len(df)):
        row = df.iloc[i]
        as_of = row["tourney_date"]
        if pd.isna(as_of):
            w_serve.append(np.nan); w_return.append(np.nan); l_serve.append(np.nan); l_return.append(np.nan)
            continue
        history = df.iloc[:i]
        ws, wr = rolling_12m_stats(history, row["winner_id"], row["surface_type"], as_of, min_matches, window_months)
        ls, lr = rolling_12m_stats(history, row["loser_id"], row["surface_type"], as_of, min_matches, window_months)
        w_serve.append(ws); w_return.append(wr); l_serve.append(ls); l_return.append(lr)
    df["winner_serve_win_pct_12m"] = w_serve
    df["winner_return_win_pct_12m"] = w_return
    df["loser_serve_win_pct_12m"] = l_serve
    df["loser_return_win_pct_12m"] = l_return
    return df


def run_pipeline(years=(2022, 2023, 2024, 2025), data_dir="data", out_path=None, attach_rolling=True):
    raw = load_and_join(years, data_dir)
    df = clean(raw)
    df = point_level_features(df)
    player_id_column(df)
    if attach_rolling:
        df = attach_rolling_stats(df)
    if out_path is None:
        out_path = os.path.join(data_dir, "wta_matches_2022_2025.csv")
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    df = run_pipeline()
    print("rows", len(df))
    print("date range", df["tourney_date"].min(), "->", df["tourney_date"].max())
    print("surface_type", df["surface_type"].value_counts().to_dict())
