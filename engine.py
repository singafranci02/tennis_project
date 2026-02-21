# Point -> game -> set -> match. 4x4 game matrix, set-score distribution, inv_logit or s/(s+r).

import json
import numpy as np
import pandas as pd
import os


def _get_player_ratings(ratings, player, surface):
    """Resolve player -> (serve, return) for a surface. Handles legacy and posterior_skills.json."""
    if "players" in ratings:
        # new format: ratings has alpha, surface_adj, players
        pl = ratings["players"].get(str(player)) or ratings["players"].get(player)
        if pl is None:
            return None, None
        surf_ratings = pl.get(surface) or pl.get("global")
        if surf_ratings is None:
            return None, None
        return surf_ratings.get("serve"), surf_ratings.get("return")
    # legacy: ratings[player][surface] = {serve, return}
    if player not in ratings:
        return None, None
    pl = ratings[player]
    surf_ratings = pl.get(surface) or pl.get("global")
    if surf_ratings is None:
        return None, None
    return surf_ratings.get("serve"), surf_ratings.get("return")


def point_win_prob(player_a, player_b, ratings, surface="global"):
    """
    P(A wins point on A's serve), P(B wins point on B's serve).
    If ratings has 'alpha' and 'surface_adj', use inv_logit(alpha + skill_serve_A - skill_return_B + surface_adj).
    Else use s_a / (s_a + r_b) with strength = exp(rating).
    """
    sa, ra = _get_player_ratings(ratings, player_a, surface)
    sb, rb = _get_player_ratings(ratings, player_b, surface)
    if sa is None or ra is None or sb is None or rb is None:
        return None, None

    if "alpha" in ratings:
        alpha = ratings["alpha"]
        adj = ratings.get("surface_adj") or {}
        adj_a = adj.get(surface, 0.0)
        # P(A wins point on A's serve)
        logit_p_a = alpha + sa - rb + adj_a
        p_a_serve = 1.0 / (1.0 + np.exp(-logit_p_a))
        logit_p_b = alpha + sb - ra + adj_a
        p_b_serve = 1.0 / (1.0 + np.exp(-logit_p_b))
        return p_a_serve, p_b_serve

    def strength(x):
        return np.exp(x)
    s_a = strength(sa)
    r_b = strength(rb)
    s_b = strength(sb)
    r_a = strength(ra)
    p_a_serve = s_a / (s_a + r_b)
    p_b_serve = s_b / (s_b + r_a)
    return p_a_serve, p_b_serve


def game_transition_matrix(p):
    """
    4x4 score states (0,15,30,40) as 0,1,2,3. Plus win(16) and lose(17).
    State k = i*4+j for server points i, returner points j.
    From (3,3) deuce: P(win) = p^2/(p^2+(1-p)^2).
    Returns 18x18 transition matrix P (row = from, col = to).
    """
    p_deuce = p ** 2 / (p ** 2 + (1 - p) ** 2)
    P = np.zeros((18, 18))
    win, lose = 16, 17
    for i in range(4):
        for j in range(4):
            k = i * 4 + j
            if i == 3 and j == 3:
                P[k, win] = p_deuce
                P[k, lose] = 1 - p_deuce
                continue
            if i == 3:
                P[k, win] = p
                if j + 1 < 4:
                    P[k, 3 * 4 + (j + 1)] = 1 - p
                else:
                    P[k, lose] = 1 - p
                continue
            if j == 3:
                P[k, lose] = 1 - p
                if i + 1 < 4:
                    P[k, (i + 1) * 4 + 3] = p
                else:
                    P[k, win] = p
                continue
            P[k, (i + 1) * 4 + j] = p
            P[k, i * 4 + (j + 1)] = 1 - p
    P[win, win] = 1.0
    P[lose, lose] = 1.0
    return P


def game_win_prob_from_matrix(p):
    """Probability of winning a game from 0-0 using the 4x4 transition matrix (with deuce)."""
    P = game_transition_matrix(p)
    # absorbing: 16=win, 17=lose. Starting state 0 = (0,0).
    # (I - Q)^-1 * R or iterate until convergence.
    Q = P[:16, :16]
    R = P[:16, 16:]
    I = np.eye(16)
    fund = np.linalg.solve(I - Q, np.eye(16))
    prob_win = (fund[0, :] @ R[:, 0])
    return float(prob_win)


def game_win_prob(p):
    """Recursive game win probability (same result as matrix form)."""
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


def set_score_distribution(player_a, player_b, ratings, surface="global"):
    """
    Probability distribution over set scores (best of 3). A is first player.
    Returns dict: "2-0", "2-1", "0-2", "1-2", and "match_win_A" = P(2-0)+P(2-1).
    """
    p_a_serve, p_b_serve = point_win_prob(player_a, player_b, ratings, surface)
    if p_a_serve is None:
        return None
    p_a_game = game_win_prob(p_a_serve)
    p_a_game_return = game_win_prob(1 - p_b_serve)
    p_a_set = set_win_prob(p_a_serve, p_b_serve, p_a_game, p_a_game_return)
    p_b_set = 1.0 - p_a_set

    # Bo3: P(A wins 2-0) = p_a_set^2
    # P(A wins 2-1) = p_a_set * p_b_set * p_a_set + p_b_set * p_a_set * p_a_set
    p_2_0 = p_a_set ** 2
    p_2_1 = 2 * p_a_set * p_b_set * p_a_set
    p_0_2 = p_b_set ** 2
    p_1_2 = 2 * p_b_set * p_a_set * p_b_set

    return {
        "2-0": p_2_0,
        "2-1": p_2_1,
        "0-2": p_0_2,
        "1-2": p_1_2,
        "match_win_A": p_2_0 + p_2_1,
    }


def match_win_prob(player_a, player_b, ratings, surface="global"):
    dist = set_score_distribution(player_a, player_b, ratings, surface)
    if dist is None:
        return None
    return dist["match_win_A"]


def load_ratings(path="data/posterior_skills.json"):
    """Load posterior_skills.json (or player_ratings.json). Prefer posterior_skills.json if both exist."""
    for p in (path, "data/posterior_skills.json", "data/player_ratings.json"):
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError("No ratings file found. Run model_core or latent_model first.")


def brier_score(df, ratings, odds_winner_col=None, odds_loser_col=None, surface_col="surface_type"):
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
        winner = row.get("winner_id") or row.get("winner_name")
        loser = row.get("loser_id") or row.get("loser_name")
        if pd.isna(winner) or pd.isna(loser):
            continue
        players_dict = ratings["players"] if "players" in ratings else ratings
        if winner not in players_dict or loser not in players_dict:
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
        if pred is None:
            continue
        preds.append(pred)
        outcomes.append(1.0)
        implied.append(imp)
    if not preds:
        return {"brier_score": None, "n": 0}
    preds = np.array(preds)
    outcomes = np.array(outcomes)
    implied = np.array(implied)
    return {
        "brier_score": float(np.mean((preds - outcomes) ** 2)),
        "brier_implied": float(np.mean((implied - outcomes) ** 2)),
        "n_matches": len(preds),
    }


if __name__ == "__main__":
    ratings = load_ratings()
    players = list(ratings.get("players", ratings).keys())
    if len(players) >= 2:
        a, b = players[0], players[1]
        pa, pb = point_win_prob(a, b, ratings)
        if pa is not None:
            dist = set_score_distribution(a, b, ratings)
            print(f"{a} vs {b}: point (A serve) {pa:.3f}, (B serve) {pb:.3f}")
            print("set scores", dist)
