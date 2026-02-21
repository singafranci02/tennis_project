# WTA match data: fetch from Sackmann repo, clean, add features, save for modeling

import pandas as pd
import numpy as np
import requests
import os

GITHUB_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"


def download_csv(year, data_dir="data"):
    fname = f"wta_matches_{year}.csv"
    path = os.path.join(data_dir, fname)
    url = f"{GITHUB_BASE}/{fname}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    os.makedirs(data_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def load_data(years=(2024, 2025), data_dir="data"):
    dfs = []
    for year in years:
        path = os.path.join(data_dir, f"wta_matches_{year}.csv")
        if not os.path.exists(path):
            download_csv(year, data_dir)
        dfs.append(pd.read_csv(path, low_memory=False))
    return pd.concat(dfs, ignore_index=True)


def clean_data(df):
    out = df.copy()
    n0 = len(out)

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
        out["tourney_date"] = pd.to_datetime(out["tourney_date"], format="%Y%m%d", errors="coerce")
        out = out[out["tourney_date"].notna()]

    return out


def engineer_features(df):
    out = df.copy()

    if "surface" in out.columns:
        out["surface_type"] = (
            out["surface"]
            .astype(str)
            .str.upper()
            .replace({"HARD": "Hard", "CLAY": "Clay", "GRASS": "Grass", "CARPET": "Carpet"})
        )
    else:
        out["surface_type"] = "Unknown"

    # Service points won % = (1stWon + 2ndWon) / svpt
    if all(c in out.columns for c in ["w_1stWon", "w_2ndWon", "w_svpt"]):
        out["winner_service_points_won_pct"] = (
            (out["w_1stWon"] + out["w_2ndWon"]) / out["w_svpt"]
        ).fillna(0) * 100
    else:
        out["winner_service_points_won_pct"] = np.nan

    if all(c in out.columns for c in ["l_1stWon", "l_2ndWon", "l_svpt"]):
        out["loser_service_points_won_pct"] = (
            (out["l_1stWon"] + out["l_2ndWon"]) / out["l_svpt"]
        ).fillna(0) * 100
    else:
        out["loser_service_points_won_pct"] = np.nan

    # Return points won % = 1 - (opponent's service points won / opponent's svpt)
    if all(c in out.columns for c in ["l_svpt", "l_1stWon", "l_2ndWon"]):
        out["winner_return_points_won_pct"] = (
            (out["l_svpt"] - out["l_1stWon"] - out["l_2ndWon"]) / out["l_svpt"]
        ).fillna(0) * 100
    else:
        out["winner_return_points_won_pct"] = np.nan

    if all(c in out.columns for c in ["w_svpt", "w_1stWon", "w_2ndWon"]):
        out["loser_return_points_won_pct"] = (
            (out["w_svpt"] - out["w_1stWon"] - out["w_2ndWon"]) / out["w_svpt"]
        ).fillna(0) * 100
    else:
        out["loser_return_points_won_pct"] = np.nan

    out["service_points_won_pct"] = out["winner_service_points_won_pct"]
    out["return_points_won_pct"] = out["winner_return_points_won_pct"]

    return out


def process_data(years=(2024, 2025), data_dir="data", out_file="wta_matches_cleaned.csv"):
    raw = load_data(years, data_dir)
    cleaned = clean_data(raw)
    final = engineer_features(cleaned)
    out_path = os.path.join(data_dir, out_file)
    final.to_csv(out_path, index=False)
    return final


if __name__ == "__main__":
    df = process_data()
    print("matches:", len(df))
    print("date range:", df["tourney_date"].min(), "->", df["tourney_date"].max())
    print(df["surface_type"].value_counts())
