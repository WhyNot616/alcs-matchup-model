"""Betting-market benchmark: match DraftKings lines (via ESPN) to games and compare with the model."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def match_odds(games, cache: dict) -> int:
    """Attach market lines to each game's pred dict. Matches on date, home and away team; doubleheaders
    are paired in start-time order with game_pk order. Returns the number of games matched."""
    by_key = defaultdict(list)
    for eid, r in cache.items():
        if not r.get("lines", {}).get("p_home_close"):
            continue
        by_key[(r["date"], r["home"], r["away"])].append((r.get("start") or "", eid, r))
    for v in by_key.values():
        v.sort()
    groups = defaultdict(list)
    for g in games:
        groups[(str(g.day), g.home, g.away)].append(g)
    n = 0
    for key, gs in groups.items():
        cands = by_key.get(key)
        if not cands:  # ESPN sometimes files late games under the next calendar day
            d = (pd.Timestamp(key[0]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            cands = by_key.get((d, key[1], key[2]))
        if not cands:
            continue
        gs = sorted(gs, key=lambda g: g.game_pk)
        if len(cands) != len(gs):  # pair by final score when counts differ
            for g in gs:
                for _, _, r in cands:
                    sc = r.get("scores") or {}
                    if str(sc.get("home")) == str(g.home_runs) and str(sc.get("away")) == str(g.away_runs):
                        _attach(g, r)
                        n += 1
                        break
            continue
        for g, (_, _, r) in zip(gs, cands):
            _attach(g, r)
            n += 1
    return n


def _attach(g, r: dict) -> None:
    L = r["lines"]
    g.pred.setdefault("baselines", {})["market"] = float(L["p_home_close"])
    if L.get("p_home_open"):
        g.pred["baselines"]["market_open"] = float(L["p_home_open"])
    g.pred["market_total"] = L.get("total_close")
    g.pred["market_ml"] = L.get("ml_close")


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def logistic(X: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Plain IRLS logistic regression. Returns coefficients and standard errors."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ b))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    return b, np.sqrt(np.diag(cov))


def model_vs_market(games) -> dict | None:
    """Does the model carry information the closing line does not?

    Fits win ~ a + b_market * logit(market) + b_model * (logit(model) - logit(market)).
    b_model near 0 means the model adds nothing beyond the market. Also scores a 50/50 logit blend.
    """
    rows = [(g.pred["p_home"], g.pred["baselines"]["market"], 1.0 if g.home_runs > g.away_runs else 0.0)
            for g in games if g.pred.get("baselines", {}).get("market") is not None]
    if len(rows) < 30:
        return None
    pm, pk, y = (np.array(c) for c in zip(*rows))
    X = np.column_stack([np.ones(len(y)), _logit(pk), _logit(pm) - _logit(pk)])
    b, se = logistic(X, y)
    blend = 1 / (1 + np.exp(-(0.5 * _logit(pm) + 0.5 * _logit(pk))))
    corr = float(np.corrcoef(pm, pk)[0, 1])
    return {"n": int(len(y)), "coef": {"intercept": float(b[0]), "market": float(b[1]), "model_minus_market": float(b[2])},
            "se": {"intercept": float(se[0]), "market": float(se[1]), "model_minus_market": float(se[2])},
            "brier_blend": float(np.mean((blend - y) ** 2)), "brier_model": float(np.mean((pm - y) ** 2)),
            "brier_market": float(np.mean((pk - y) ** 2)), "corr_model_market": corr,
            "mean_abs_gap": float(np.mean(np.abs(pm - pk)))}


def totals_check(games) -> dict | None:
    """Model expected total runs vs the market's closing total vs what happened."""
    rows = [(g.pred["exp_home"] + g.pred["exp_away"], g.pred["market_total"], g.home_runs + g.away_runs)
            for g in games if g.pred.get("market_total")]
    if len(rows) < 5:
        return None
    m, k, a = (np.array(c, float) for c in zip(*rows))
    return {"n": int(len(a)), "model_mean": float(m.mean()), "market_mean": float(k.mean()), "actual_mean": float(a.mean()),
            "rmse_model": float(np.sqrt(np.mean((m - a) ** 2))), "rmse_market": float(np.sqrt(np.mean((k - a) ** 2))),
            "corr_model_market": float(np.corrcoef(m, k)[0, 1]) if np.std(m) > 0 and np.std(k) > 0 else None}
