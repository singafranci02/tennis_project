"""
Ingram-style hierarchical Bayesian point-level model.
Likelihood: P(win point on serve) = inv_logit(alpha + skill_serve_A - skill_return_B + surface_adj).
Skills evolve over time via Gaussian Random Walk; surface_adj with shrinkage.
Exports posterior means to JSON for the engine.
"""

import json
import os
import numpy as np
import pandas as pd

try:
    import pymc as pm
    from pymc import Binomial, Normal, invlogit
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


def load_match_data(path="data/wta_matches_2022_2025.csv", top_n=250):
    df = pd.read_csv(path, low_memory=False)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.dropna(subset=["tourney_date"])

    # player_id: prefer winner_id/loser_id
    if "winner_id" not in df.columns:
        df["winner_id"] = df["winner_name"].astype(str).str.strip()
    if "loser_id" not in df.columns:
        df["loser_id"] = df["loser_name"].astype(str).str.strip()

    # point-level counts
    if "w_svpt" in df.columns and "w_1stWon" in df.columns and "w_2ndWon" in df.columns:
        df["winner_serve_won"] = (df["w_1stWon"] + df["w_2ndWon"]).fillna(0).astype(int)
    else:
        df["winner_serve_won"] = np.nan
    if "l_svpt" in df.columns and "l_1stWon" in df.columns and "l_2ndWon" in df.columns:
        df["loser_serve_won"] = (df["l_1stWon"] + df["l_2ndWon"]).fillna(0).astype(int)
    else:
        df["loser_serve_won"] = np.nan

    df["winner_svpt"] = df["w_svpt"].fillna(0).astype(int)
    df["loser_svpt"] = df["l_svpt"].fillna(0).astype(int)

    # surface
    if "surface_type" not in df.columns and "surface" in df.columns:
        df["surface_type"] = df["surface"].astype(str).str.upper().replace(
            {"HARD": "Hard", "CLAY": "Clay", "GRASS": "Grass", "CARPET": "Carpet"}
        )
    df["surface_type"] = df["surface_type"].fillna("Hard").astype(str)

    # period: monthly bins
    df["period"] = (df["tourney_date"].dt.year - 2022) * 12 + df["tourney_date"].dt.month

    # top players by match count
    all_players = pd.concat([
        df["winner_id"].value_counts(),
        df["loser_id"].value_counts()
    ]).groupby(level=0).sum().sort_values(ascending=False)
    top_ids = set(all_players.head(top_n).index)
    df = df[df["winner_id"].isin(top_ids) & df["loser_id"].isin(top_ids)].copy()

    # drop rows missing required cols
    df = df.dropna(subset=["winner_serve_won", "loser_serve_won", "winner_svpt", "loser_svpt"])
    df = df[(df["winner_svpt"] > 0) & (df["loser_svpt"] > 0)]

    players = sorted(set(df["winner_id"].unique()) | set(df["loser_id"].unique()))
    surfaces = sorted(df["surface_type"].unique())
    periods = sorted(df["period"].unique())
    player_to_idx = {p: i for i, p in enumerate(players)}
    surface_to_idx = {s: i for i, s in enumerate(surfaces)}
    period_to_idx = {t: i for i, t in enumerate(periods)}

    winner_idx = []
    loser_idx = []
    period_idx = []
    surface_idx = []
    winner_svpt = []
    winner_serve_won = []
    loser_svpt = []
    loser_serve_won = []

    for _, row in df.iterrows():
        winner_idx.append(player_to_idx[row["winner_id"]])
        loser_idx.append(player_to_idx[row["loser_id"]])
        period_idx.append(period_to_idx[row["period"]])
        surface_idx.append(surface_to_idx[row["surface_type"]])
        winner_svpt.append(int(row["winner_svpt"]))
        winner_serve_won.append(int(row["winner_serve_won"]))
        loser_svpt.append(int(row["loser_svpt"]))
        loser_serve_won.append(int(row["loser_serve_won"]))

    return {
        "winner_idx": np.array(winner_idx),
        "loser_idx": np.array(loser_idx),
        "period_idx": np.array(period_idx),
        "surface_idx": np.array(surface_idx),
        "winner_svpt": np.array(winner_svpt),
        "winner_serve_won": np.array(winner_serve_won),
        "loser_svpt": np.array(loser_svpt),
        "loser_serve_won": np.array(loser_serve_won),
        "players": players,
        "surfaces": surfaces,
        "periods": periods,
        "n_players": len(players),
        "n_surfaces": len(surfaces),
        "n_periods": len(periods),
        "n_matches": len(winner_idx),
    }


def build_model(data):
    """PyMC model: inv_logit likelihood, Gaussian RW for skills, surface_adj with shrinkage."""
    n_players = data["n_players"]
    n_periods = data["n_periods"]
    n_surfaces = data["n_surfaces"]

    coords = {
        "players": np.arange(n_players),
        "periods": np.arange(n_periods),
        "surfaces": np.arange(n_surfaces),
        "matches": np.arange(data["n_matches"]),
    }

    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        sigma_rw_serve = pm.HalfNormal("sigma_rw_serve", sigma=0.3)
        sigma_rw_return = pm.HalfNormal("sigma_rw_return", sigma=0.3)

        skill_serve_0 = pm.Normal("skill_serve_0", mu=0, sigma=1, dims="players")
        skill_return_0 = pm.Normal("skill_return_0", mu=0, sigma=1, dims="players")
        skill_serve_delta = pm.Normal("skill_serve_delta", mu=0, sigma=sigma_rw_serve, dims=("players", "periods"))
        skill_return_delta = pm.Normal("skill_return_delta", mu=0, sigma=sigma_rw_return, dims=("players", "periods"))

        # skill[t] = skill_0 + cumsum(delta)[:t]; cumsum has shape (players, n_periods), take first n_periods cols
        cum_serve = pm.math.concatenate([
            pm.math.zeros((n_players, 1)),
            pm.math.cumsum(skill_serve_delta[:, : n_periods - 1], axis=1)
        ], axis=1)
        cum_return = pm.math.concatenate([
            pm.math.zeros((n_players, 1)),
            pm.math.cumsum(skill_return_delta[:, : n_periods - 1], axis=1)
        ], axis=1)
        skill_serve = pm.Deterministic("skill_serve", skill_serve_0[:, None] + cum_serve[:, :n_periods])
        skill_return = pm.Deterministic("skill_return", skill_return_0[:, None] + cum_return[:, :n_periods])

        surface_adj_rest = pm.Normal("surface_adj_rest", mu=0, sigma=0.5, shape=(n_surfaces - 1,))
        surface_adj = pm.Deterministic("surface_adj", pm.math.concatenate([[0.0], surface_adj_rest]))

        w_idx = data["winner_idx"]
        l_idx = data["loser_idx"]
        t_idx = data["period_idx"]
        s_idx = data["surface_idx"]

        p_winner_serve = pm.math.invlogit(
            alpha + skill_serve[w_idx, t_idx] - skill_return[l_idx, t_idx] + surface_adj[s_idx]
        )
        p_loser_serve = pm.math.invlogit(
            alpha + skill_serve[l_idx, t_idx] - skill_return[w_idx, t_idx] + surface_adj[s_idx]
        )

        pm.Binomial("obs_winner_serve", n=data["winner_svpt"], p=p_winner_serve, observed=data["winner_serve_won"])
        pm.Binomial("obs_loser_serve", n=data["loser_svpt"], p=p_loser_serve, observed=data["loser_serve_won"])

    return model


def fit_and_export(data, output_path="data/posterior_skills.json", draws=500, tune=500):
    if not PYMC_AVAILABLE:
        raise RuntimeError("PyMC not installed. pip install pymc")

    model = build_model(data)
    with model:
        idata = pm.sample(draws=draws, tune=tune, target_accept=0.9, return_inferencedata=True)

    skill_serve = idata.posterior["skill_serve"].mean(dim=("chain", "draw")).values
    skill_return = idata.posterior["skill_return"].mean(dim=("chain", "draw")).values
    surface_adj = idata.posterior["surface_adj"].mean(dim=("chain", "draw")).values
    alpha = float(idata.posterior["alpha"].mean(dim=("chain", "draw")).values)

    t_last = data["n_periods"] - 1
    players = data["players"]
    surfaces = data["surfaces"]

    # surface_adj may be 0-dim or 1-dim
    if surface_adj.ndim == 0:
        surface_adj = np.array([float(surface_adj)])
    else:
        surface_adj = np.asarray(surface_adj).ravel()

    out = {
        "alpha": alpha,
        "surface_adj": {s: float(surface_adj[i]) for i, s in enumerate(surfaces)},
        "players": {},
    }
    for i, pid in enumerate(players):
        s_serve = float(skill_serve[i, t_last])
        s_return = float(skill_return[i, t_last])
        out["players"][str(pid)] = {"global": {"serve": s_serve, "return": s_return}}
        for j, surf in enumerate(surfaces):
            adj = float(surface_adj[j]) if j < len(surface_adj) else 0.0
            out["players"][str(pid)][surf] = {"serve": s_serve + adj, "return": s_return}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    data_path = "data/wta_matches_2022_2025.csv"
    if not os.path.exists(data_path):
        raise SystemExit("Run data_loader.py first to create " + data_path)

    data = load_match_data(data_path, top_n=200)
    print("matches", data["n_matches"], "players", data["n_players"], "periods", data["n_periods"])

    out = fit_and_export(data, draws=300, tune=300)
    print("posterior_skills.json written")


if __name__ == "__main__":
    main()
