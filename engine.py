# Point -> game -> set -> match win prob from serve/return strengths. Brier vs odds.

import json
import numpy as np
import pandas as pd
import os

# P(A wins point on A's serve) = s_a / (s_a + r_b)
def point_win_prob(player_a, player_b, ratings, surface="global"):
    if surface not in ratings.get(player_a, {}):
        surface = "global"
    # ratings are on logit scale; we need positive strengths
    def strength(x):
        return np.exp(x)
    s_a = strength(ratings[player_a][surface]["serve"])
    r_a = strength(ratings[player_a][surface]["return"])
    s_b = strength(ratings[player_b][surface]["serve"])
    r_b = strength(ratings[player_b][surface]["return"])
    p_a_serve = s_a / (s_a + r_b)
    p_b_serve = s_b / (s_b + r_a)
    return p_a_serve, p_b_serve


def game_win_prob(p):
    memo = {}

    def P(a, b):
        if (a, b) in memo:
            return memo[(a, b)]
        if a >= 4 and a - b >= 2:
            return 1.0
        if b >= 4 and b - a >= 2:
            return 0.0
        if a == 4 and b == 3:
            r = p * 1.0 + (1 - p) * P(3, 3)
        elif a == 3 and b == 4:
            r = p * P(3, 3) + (1 - p) * 0.0
        elif a == 3 and b == 3:
            r = p ** 2 / (p ** 2 + (1 - p) ** 2)
        else:
            r = p * P(a + 1, b) + (1 - p) * P(a, b + 1)
        memo[(a, b)] = r
        return r

    return P(0, 0)


def tiebreak_win_prob(p_a_serve, p_b_serve):
    memo = {}

    def P(a, b, a_serves_first):
        if (a, b, a_serves_first) in memo:
            return memo[(a, b, a_serves_first)]
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0
        total = a + b
        a_serves = (total % 2 == 0) if a_serves_first else (total % 2 == 1)
        q = p_a_serve if a_serves else (1 - p_b_serve)
        r = q * P(a + 1, b, a_serves_first) + (1 - q) * P(a, b + 1, a_serves_first)
        memo[(a, b, a_serves_first)] = r
        return r

    return P(0, 0, True)


def set_win_prob(p_a_serve, p_b_serve, p_a_game, p_a_game_when_b_serves):
    memo = {}

    def P(ga, gb, a_serves):
        if (ga, gb, a_serves) in memo:
            return memo[(ga, gb, a_serves)]
        if ga >= 6 and ga - gb >= 2:
            return 1.0
        if gb >= 6 and gb - ga >= 2:
            return 0.0
        if ga == 6 and gb == 6:
            return tiebreak_win_prob(p_a_serve, p_b_serve)
        q = p_a_game if a_serves else p_a_game_when_b_serves
        r = q * P(ga + 1, gb, not a_serves) + (1 - q) * P(ga, gb + 1, not a_serves)
        memo[(ga, gb, a_serves)] = r
        return r

    return P(0, 0, True)


def match_win_prob(player_a, player_b, ratings, surface="global"):
    p_a_serve, p_b_serve = point_win_prob(player_a, player_b, ratings, surface)
    p_a_game = game_win_prob(p_a_serve)
    p_a_game_return = game_win_prob(1 - p_b_serve)
    p_a_set = set_win_prob(p_a_serve, p_b_serve, p_a_game, p_a_game_return)
    # best of 3
    return p_a_set ** 2 + 2 * p_a_set * (1 - p_a_set) * p_a_set


def load_ratings(path="data/player_ratings.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run latent_model first. Missing: {path}")
    with open(path) as f:
        return json.load(f)


def brier_score(df, ratings, odds_winner_col=None, odds_loser_col=None, surface_col="surface_type"):
    """Brier = E[(pred - outcome)^2]. Lower is better. Compare model pred to implied prob from odds."""
    if odds_winner_col is None:
        for c in ["B365W", "PSW", "winner_odds", "w_odds"]:
            if c in df.columns:
                odds_winner_col = c
                break
    if odds_loser_col is None:
        for c in ["B365L", "PSL", "loser_odds", "l_odds"]:
            if c in df.columns:
                odds_loser_col = c
                break
    if odds_winner_col is None or odds_loser_col is None:
        return {"brier_score": None, "n": 0, "message": "no odds columns"}

    preds, outcomes, implied = [], [], []
    for _, row in df.iterrows():
        winner = row.get("winner_name")
        loser = row.get("loser_name")
        if pd.isna(winner) or pd.isna(loser):
            continue
        if winner not in ratings or loser not in ratings:
            continue
        ow = row.get(odds_winner_col)
        ol = row.get(odds_loser_col)
        if pd.isna(ow) or pd.isna(ol):
            continue
        try:
            ow, ol = float(ow), float(ol)
        except (ValueError, TypeError):
            continue
        imp = (1.0 / ow) / (1.0 / ow + 1.0 / ol)
        surf = row.get(surface_col, "global")
        pred = match_win_prob(winner, loser, ratings, surf)
        preds.append(pred)
        outcomes.append(1.0)
        implied.append(imp)
    if not preds:
        return {"brier_score": None, "n": 0}
    preds = np.array(preds)
    outcomes = np.array(outcomes)
    implied = np.array(implied)
    brier_model = np.mean((preds - outcomes) ** 2)
    brier_implied = np.mean((implied - outcomes) ** 2)
    return {
        "brier_score": float(brier_model),
        "brier_implied": float(brier_implied),
        "n_matches": len(preds),
    }


if __name__ == "__main__":
    ratings = load_ratings()
    players = list(ratings.keys())
    if len(players) >= 2:
        a, b = players[0], players[1]
        pa, pb = point_win_prob(a, b, ratings)
        pm = match_win_prob(a, b, ratings)
        print(f"{a} vs {b}: point (A serve) {pa:.3f}, (B serve) {pb:.3f}, match(A) {pm:.3f}")

    if os.path.exists("data/wta_matches_cleaned.csv"):
        df = pd.read_csv("data/wta_matches_cleaned.csv")
        res = brier_score(df, ratings)
        print("Brier:", res)
