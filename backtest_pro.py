"""
Backtest model vs 2025 closing odds: Brier, Log Loss, segment by tourney tier,
fractional Kelly staking, equity curve, Sharpe, max drawdown.
"""

import os
import numpy as np
import pandas as pd

from engine import load_ratings, match_win_prob

ODDS_W = "B365W"
ODDS_L = "B365L"
SURFACE_COL = "surface_type"
TOURNEY_LEVEL_COL = "tourney_level"
KELLY_FRAC = 0.25


def implied_prob(odds_w, odds_l):
    iw = 1.0 / float(odds_w)
    il = 1.0 / float(odds_l)
    tot = iw + il
    return iw / tot, il / tot


def log_loss(pred, outcome):
    eps = 1e-12
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean(outcome * np.log(pred) + (1 - outcome) * np.log(1 - pred))


def load_2025_matches(path="data/wta_matches_2022_2025.csv"):
    df = pd.read_csv(path, low_memory=False)
    if "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
        df = df[df["tourney_date"].dt.year == 2025]
    else:
        df = df[df["year"] == 2025] if "year" in df.columns else df
    return df.sort_values("tourney_date" if "tourney_date" in df.columns else df.index).reset_index(drop=True)


def backtest(df, ratings, odds_w_col=None, odds_l_col=None, surface_col=SURFACE_COL,
             tourney_level_col=TOURNEY_LEVEL_COL, kelly_frac=KELLY_FRAC):
    if odds_w_col is None:
        for c in ["B365W", "PSW", "winner_odds", "w_odds"]:
            if c in df.columns:
                odds_w_col = c
                break
    if odds_l_col is None:
        for c in ["B365L", "PSL", "loser_odds", "l_odds"]:
            if c in df.columns:
                odds_l_col = c
                break
    if odds_w_col is None or odds_l_col is None:
        return None
    players = ratings.get("players", ratings)

    rows = []
    for _, row in df.iterrows():
        winner = row.get("winner_id") or row.get("winner_name")
        loser = row.get("loser_id") or row.get("loser_name")
        if pd.isna(winner) or pd.isna(loser) or winner not in players or loser not in players:
            continue
        ow = row.get(odds_w_col)
        ol = row.get(odds_l_col)
        if pd.isna(ow) or pd.isna(ol):
            continue
        try:
            ow, ol = float(ow), float(ol)
        except (ValueError, TypeError):
            continue
        imp_w, imp_l = implied_prob(ow, ol)
        surf = row.get(surface_col, "global")
        pred_w = match_win_prob(winner, loser, ratings, surf)
        if pred_w is None:
            continue
        outcome = 1.0  # winner won
        # Kelly: stake on winner = f * (pred * odds - 1) / (odds - 1) when positive edge
        edge_w = pred_w - imp_w
        if pred_w * ow > 1 and ow > 1:
            q = (pred_w * ow - 1) / (ow - 1)
            stake = kelly_frac * q
        else:
            stake = 0.0
        if stake > 0:
            pnl = stake * (ow - 1) if outcome == 1.0 else -stake
        else:
            pnl = 0.0
        level = row.get(tourney_level_col) or "Other"
        rows.append({
            "pred": pred_w,
            "implied": imp_w,
            "outcome": outcome,
            "odds_w": ow,
            "stake": stake,
            "pnl": pnl,
            "tourney_level": level,
        })

    if not rows:
        return None
    out = pd.DataFrame(rows)
    out["cum_pnl"] = out["pnl"].cumsum()
    return out


def metrics(bt):
    if bt is None or len(bt) == 0:
        return {}
    pred = bt["pred"].values
    outcome = bt["outcome"].values
    implied = bt["implied"].values
    brier = np.mean((pred - outcome) ** 2)
    brier_impl = np.mean((implied - outcome) ** 2)
    ll = log_loss(pred, outcome)
    ll_impl = log_loss(implied, outcome)
    returns = bt["pnl"].values
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(len(returns))) if np.std(returns) > 0 else 0.0
    cum = bt["cum_pnl"].values
    run_max = np.maximum.accumulate(cum)
    drawdown = run_max - cum
    max_dd = float(np.max(drawdown))
    return {
        "brier": brier,
        "brier_implied": brier_impl,
        "log_loss": ll,
        "log_loss_implied": ll_impl,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_pnl": float(cum[-1]),
        "n": len(bt),
    }


def by_segment(bt, tourney_level_col="tourney_level"):
    if bt is None or tourney_level_col not in bt.columns:
        return {}
    seg = {}
    for level in bt[tourney_level_col].unique():
        sub = bt[bt[tourney_level_col] == level]
        seg[str(level)] = metrics(sub)
    return seg


def run(data_path="data/wta_matches_2022_2025.csv", ratings_path="data/posterior_skills.json",
        kelly_frac=KELLY_FRAC, plot=True):
    if not os.path.exists(data_path):
        print("Missing", data_path, "- run data_loader.py first")
        return
    ratings = load_ratings(ratings_path)
    df = load_2025_matches(data_path)
    if len(df) == 0:
        print("No 2025 matches in", data_path)
        return

    bt = backtest(df, ratings, kelly_frac=kelly_frac)
    if bt is None:
        print("No backtest rows (missing odds or players)")
        return

    m = metrics(bt)
    print("--- Backtest 2025 ---")
    print("n_matches", m["n"])
    print("Brier (model)", m["brier"])
    print("Brier (implied)", m["brier_implied"])
    print("Log Loss (model)", m["log_loss"])
    print("Log Loss (implied)", m["log_loss_implied"])
    print("Sharpe", m["sharpe"])
    print("Max Drawdown", m["max_drawdown"])
    print("Total PnL", m["total_pnl"])

    seg = by_segment(bt)
    if seg:
        print("\n--- By tournament tier ---")
        for level, sm in seg.items():
            print(f"  {level}: Brier={sm['brier']:.4f} LogLoss={sm['log_loss']:.4f} n={sm['n']}")

    if plot and len(bt) > 0:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(bt["cum_pnl"].values, label="Cumulative PnL")
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_xlabel("Match (time order)")
            ax.set_ylabel("Cumulative PnL")
            ax.set_title("Equity curve (fractional Kelly)")
            ax.legend()
            out_plot = "data/equity_curve.png"
            os.makedirs("data", exist_ok=True)
            plt.savefig(out_plot, dpi=120)
            plt.close()
            print("\nPlot saved to", out_plot)
        except Exception as e:
            print("Plot failed:", e)

    return {"metrics": m, "by_segment": seg, "backtest_df": bt}


if __name__ == "__main__":
    run(plot=True)
