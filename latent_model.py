# Hierarchical Bradley-Terry: estimate serve/return strengths, surface-specific with shrinkage

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import json
import os


def _top_players(df, n=200):
    w = df["winner_name"].value_counts()
    l = df["loser_name"].value_counts()
    combined = w.add(l, fill_value=0).sort_values(ascending=False)
    return set(combined.head(n).index)


def prepare_match_list(df, top_n=200):
    top = _top_players(df, top_n)
    sub = df[df["winner_name"].isin(top) & df["loser_name"].isin(top)].copy()

    players = sorted(set(sub["winner_name"].unique()) | set(sub["loser_name"].unique()))
    player_to_idx = {p: i for i, p in enumerate(players)}
    surfaces = sorted(sub["surface_type"].dropna().unique())
    surface_to_idx = {s: i for i, s in enumerate(surfaces)}

    matches = []
    for _, row in sub.iterrows():
        surface = row.get("surface_type")
        if surface not in surface_to_idx:
            continue
        w_serve = row.get("winner_service_points_won_pct")
        l_serve = row.get("loser_service_points_won_pct")
        w_ret = row.get("winner_return_points_won_pct")
        l_ret = row.get("loser_return_points_won_pct")
        w_svpt = row.get("w_svpt")
        l_svpt = row.get("l_svpt")
        if pd.isna(w_serve) or pd.isna(l_serve) or pd.isna(w_ret) or pd.isna(l_ret):
            continue
        if pd.isna(w_svpt) or pd.isna(l_svpt):
            continue
        matches.append({
            "winner_idx": player_to_idx[row["winner_name"]],
            "loser_idx": player_to_idx[row["loser_name"]],
            "surface_idx": surface_to_idx[surface],
            "winner_serve_pct": w_serve / 100.0,
            "loser_serve_pct": l_serve / 100.0,
            "winner_return_pct": w_ret / 100.0,
            "loser_return_pct": l_ret / 100.0,
            "winner_svpt": int(w_svpt),
            "loser_svpt": int(l_svpt),
        })

    return {
        "matches": matches,
        "players": players,
        "surfaces": surfaces,
        "player_to_idx": player_to_idx,
        "surface_to_idx": surface_to_idx,
        "n_players": len(players),
        "n_surfaces": len(surfaces),
    }


def unpack(params, n_players, n_surfaces):
    i = 0
    global_serve = params[i : i + n_players]
    i += n_players
    global_return = params[i : i + n_players]
    i += n_players
    surface_serve = params[i : i + n_players * n_surfaces].reshape(n_players, n_surfaces)
    i += n_players * n_surfaces
    surface_return = params[i : i + n_players * n_surfaces].reshape(n_players, n_surfaces)
    return global_serve, global_return, surface_serve, surface_return


def nll(params, data, shrinkage=0.1):
    matches = data["matches"]
    n_players = data["n_players"]
    n_surfaces = data["n_surfaces"]
    g_s, g_r, s_s, s_r = unpack(params, n_players, n_surfaces)

    ll = 0.0
    for m in matches:
        wi, li, si = m["winner_idx"], m["loser_idx"], m["surface_idx"]
        pw = expit(s_s[wi, si] - s_r[li, si])
        pl = expit(s_s[li, si] - s_r[wi, si])
        nw, nl = m["winner_svpt"], m["loser_svpt"]
        ow = m["winner_serve_pct"] * nw
        ol = m["loser_serve_pct"] * nl
        if nw > 0:
            ll += ow * np.log(pw + 1e-10) + (nw - ow) * np.log(1 - pw + 1e-10)
        if nl > 0:
            ll += ol * np.log(pl + 1e-10) + (nl - ol) * np.log(1 - pl + 1e-10)

    # shrink surface toward global
    for p in range(n_players):
        for s in range(n_surfaces):
            ll -= shrinkage * ((s_s[p, s] - g_s[p]) ** 2 + (s_r[p, s] - g_r[p]) ** 2)
    ll -= 0.01 * np.sum(params ** 2)

    return -ll


def fit(data, shrinkage=0.1, max_iter=1000):
    n = data["n_players"]
    s = data["n_surfaces"]
    n_params = 2 * n + 2 * n * s
    x0 = np.random.normal(0, 0.1, n_params)
    bounds = [(-5, 5)] * n_params
    res = minimize(
        lambda p: nll(p, data, shrinkage),
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "disp": True},
    )
    return res.x


def ratings_dict(fitted_params, data):
    g_s, g_r, s_s, s_r = unpack(
        fitted_params, data["n_players"], data["n_surfaces"]
    )
    players = data["players"]
    surfaces = data["surfaces"]
    out = {}
    for i, name in enumerate(players):
        out[name] = {"global": {"serve": float(g_s[i]), "return": float(g_r[i])}}
        for j, surf in enumerate(surfaces):
            out[name][surf] = {
                "serve": float(s_s[i, j]),
                "return": float(s_r[i, j]),
            }
    return out


def main():
    path = "data/wta_matches_cleaned.csv"
    if not os.path.exists(path):
        raise SystemExit(f"Run data_manager first. Missing: {path}")

    df = pd.read_csv(path)
    data = prepare_match_list(df, top_n=200)
    params = fit(data)
    ratings = ratings_dict(params, data)

    os.makedirs("data", exist_ok=True)
    with open("data/player_ratings.json", "w") as f:
        json.dump(ratings, f, indent=2)

    # quick summary
    gs = [r["global"]["serve"] for r in ratings.values()]
    gr = [r["global"]["return"] for r in ratings.values()]
    print("global serve  mean %.3f std %.3f" % (np.mean(gs), np.std(gs)))
    print("global return mean %.3f std %.3f" % (np.mean(gr), np.std(gr)))
    top_s = sorted(ratings.items(), key=lambda x: x[1]["global"]["serve"], reverse=True)[:5]
    top_r = sorted(ratings.items(), key=lambda x: x[1]["global"]["return"], reverse=True)[:5]
    print("top serve:", [p[0] for p in top_s])
    print("top return:", [p[0] for p in top_r])


if __name__ == "__main__":
    main()
