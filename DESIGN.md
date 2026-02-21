# WTA Enterprise Quant Model – Design Notes

## Pipeline

1. **data_loader.py** – Ingest 2022–2025 Sackmann WTA data, point-level serve/return stats, rolling 12‑month averages with a minimum match threshold per surface and **no look-ahead** (stats as of the day before each match).
2. **model_core.py** – Hierarchical Bayesian point-level model (PyMC): `P(win point on serve) = inv_logit(α + skill_serve_A − skill_return_B + surface_adj)`. Gaussian Random Walk over time for skills; surface effects shrunk toward zero. Exports posterior means to `data/posterior_skills.json`.
3. **engine.py** – Markov chain: point → game (4×4 transition matrix + deuce) → set (incl. tiebreak) → match. Full **set-score distribution** (2‑0, 2‑1, 0‑2, 1‑2) for set betting.
4. **backtest_pro.py** – Backtest vs 2025 closing odds: Brier, Log Loss, segmentation by tournament tier, fractional Kelly, equity curve, Sharpe, max drawdown.

---

## Three Concepts for Interviews

### 1. Bayesian shrinkage

Surface-specific estimates are **shrunk toward the player’s global mean** (and league-wide surface effects toward zero). That avoids overreacting to a single good run on one surface (e.g. one strong grass tournament) and stabilizes estimates when there are few matches on that surface.

### 2. Point-level generative model

We do **not** model “who wins the match” directly. We model **who wins each point** (on serve). Match, set, and game probabilities are then derived via the Markov chain. That keeps the model generative at the point level and allows **live/in-play** use: probabilities can be updated after every point.

### 3. WTA volatility thesis

WTA serve‑hold variance is higher than on the ATP. A **point-level** model that separates serve and return skill is better suited to capture that and to find **mispriced underdogs**, especially when return strength is underrated by the market.

---

## Stress test: “How do you handle a player returning from a long injury layoff?”

**Answer:** In a Bayesian setup we can **increase the variance (uncertainty)** of that player’s latent skill (e.g. wider prior or larger step variance in the Random Walk). The posterior then stays **wider** for that player, so the model is more cautious. Under a **Kelly** (or fractional Kelly) staking rule, that uncertainty **reduces bet size** until new data narrows the posterior. Optionally we can flag “low match count in the last N months” and scale up prior variance for those players (e.g. in a future extension in `model_core`).
