"""Plate-appearance outcome model and backtest.

Outcome probabilities for a batter vs a pitcher come from the multinomial odds-ratio method:

    p(x) is proportional to  batter_rate(x) * pitcher_rate(x) / league_rate(x)

with each rate split by platoon (batter rates vs the pitcher's hand, pitcher rates vs the batter's
side) and regressed toward league average using outcome-specific stabilization constants.
An optional pitch-mix tilt moves probability between "good for the hitter" outcomes (BB, hits) and
"good for the pitcher" outcomes (K, outs) in proportion to the arsenal edge from matchups.py.
The backtest decides whether that tilt earns its place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import OUTCOMES, WOBA_W, recency_weights
from .matchups import build_profiles, pair_edges

GOOD = np.array([o in ("BB", "1B", "2B", "3B", "HR") for o in OUTCOMES])


@dataclass
class Rates:
    league: dict[tuple[str, str], np.ndarray]     # (stand, p_throws) -> probs
    batter: dict[tuple[int, str], np.ndarray]     # (batter, p_throws) -> probs
    pitcher: dict[tuple[int, str], np.ndarray]    # (pitcher, stand) -> probs
    batter_n: dict[tuple[int, str], float]
    pitcher_n: dict[tuple[int, str], float]
    stuff: dict[int, float] = field(default_factory=dict)   # pitcher -> stuff runs saved per 100 (stuff.py)
    gamma: float = 0.0                                        # weight of the stuff tilt (chosen in backtest)


def _counts(pa: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    t = pa.pivot_table(index=keys, columns="outcome", values="w", aggfunc="sum", fill_value=0.0, observed=True)
    for o in OUTCOMES:
        if o not in t.columns:
            t[o] = 0.0
    return t[OUTCOMES]


def build_rates(pa: pd.DataFrame, shrink_pa: dict, half_life: float = 0,
                as_of: pd.Timestamp | None = None) -> Rates:
    d = pa[pa["outcome"].isin(OUTCOMES)].copy()
    d["stand"] = d["stand"].astype(str)
    d["p_throws"] = d["p_throws"].astype(str)
    d["w"] = recency_weights(d["game_date"], half_life, as_of)
    if "bw" in d.columns:
        d["w"] = d["w"] * d["bw"].to_numpy()
    k = np.array([shrink_pa[o] for o in OUTCOMES], dtype=float)

    lg_c = _counts(d, ["stand", "p_throws"])
    league = {idx: (row / row.sum()).to_numpy() for idx, row in lg_c.iterrows()}
    lg_stand = _counts(d, ["stand"])
    lg_stand = {i: (r / r.sum()).to_numpy() for i, r in lg_stand.iterrows()}
    lg_side_p = _counts(d, ["p_throws"])
    lg_side_p = {i: (r / r.sum()).to_numpy() for i, r in lg_side_p.iterrows()}

    def shrink(c: np.ndarray, prior: np.ndarray, kk: np.ndarray) -> np.ndarray:
        n = c.sum()
        r = (c + kk * prior) / (n + kk)
        return r / r.sum()

    # batters: overall (toward league for their side) then by pitcher hand (toward overall x platoon ratio)
    bat_all = _counts(d, ["batter", "stand"])
    bat_hand = _counts(d, ["batter", "stand", "p_throws"])
    batter, batter_n = {}, {}
    overall = {}
    for (b, s), row in bat_all.iterrows():
        overall[(b, s)] = shrink(row.to_numpy(), lg_stand[s], k)
    for (b, s, h), row in bat_hand.iterrows():
        ratio = league.get((s, h), lg_stand[s]) / lg_stand[s]
        prior = overall[(b, s)] * ratio
        prior = prior / prior.sum()
        r = shrink(row.to_numpy(), prior, k * 1.5)
        key = (int(b), h)
        # switch hitters appear under two stands; keep the larger sample
        if key not in batter or row.sum() > batter_n[key]:
            batter[key], batter_n[key] = r, float(row.sum())
    # pitchers: overall then by batter side
    pit_all = _counts(d, ["pitcher", "p_throws"])
    pit_side = _counts(d, ["pitcher", "p_throws", "stand"])
    p_over = {}
    for (p, h), row in pit_all.iterrows():
        p_over[(p, h)] = shrink(row.to_numpy(), lg_side_p[h], k)
    pitcher, pitcher_n = {}, {}
    for (p, h, s), row in pit_side.iterrows():
        ratio = league.get((s, h), lg_side_p[h]) / lg_side_p[h]
        prior = p_over[(p, h)] * ratio
        prior = prior / prior.sum()
        pitcher[(int(p), s)] = shrink(row.to_numpy(), prior, k * 1.5)
        pitcher_n[(int(p), s)] = float(row.sum())
    return Rates(league, batter, pitcher, batter_n, pitcher_n)


def matchup_probs(rates: Rates, batter: int, bat_side: str, pitcher: int, p_throws: str,
                  edge: float = 0.0, lam: float = 0.0, park: np.ndarray | None = None) -> np.ndarray:
    lg = rates.league.get((bat_side, p_throws))
    if lg is None:
        lg = np.mean(list(rates.league.values()), axis=0)
    b = rates.batter.get((int(batter), p_throws), lg)
    p = rates.pitcher.get((int(pitcher), bat_side), lg)
    x = b * p / lg
    if lam and edge:
        x = x * np.where(GOOD, np.exp(lam * edge), np.exp(-lam * edge))
    if rates.gamma:
        sv = rates.stuff.get(int(pitcher))
        if sv:
            x = x * np.where(GOOD, np.exp(-rates.gamma * sv), np.exp(rates.gamma * sv))
    if park is not None:
        x = x * park
    return x / x.sum()


def batter_side(batter_bats: str, p_throws: str) -> str:
    """Switch hitters bat opposite the pitcher's hand."""
    if batter_bats == "S":
        return "L" if p_throws == "R" else "R"
    return batter_bats


def expected_woba(probs: np.ndarray) -> float:
    w = np.array([WOBA_W[o] for o in OUTCOMES])
    return float(probs @ w)


# ---------------------------------------------------------------- backtest
def _logloss(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def _predict(rates: Rates, test: pd.DataFrame, edges: np.ndarray | None, lam: float) -> np.ndarray:
    out = np.empty((len(test), len(OUTCOMES)))
    for i, (b, s, p, h) in enumerate(test[["batter", "stand", "pitcher", "p_throws"]].itertuples(index=False)):
        e = float(edges[i]) if edges is not None else 0.0
        out[i] = matchup_probs(rates, b, str(s), p, str(h), e, lam)
    return out


def backtest(pitches: pd.DataFrame, pa: pd.DataFrame, cfg_model: dict, verbose: bool = True) -> dict:
    """Train on PAs before the split date, test on PAs after it.

    Models compared on multiclass log loss (lower is better):
      league   : league outcome rates by platoon only
      oddsratio: batter and pitcher rates combined (talent + platoon)
      + mix    : odds ratio plus the pitch-mix tilt, with lambda chosen on the last 30 days of training
    """
    split = pd.Timestamp(cfg_model["backtest_split"])
    hl = cfg_model.get("recent_half_life_days", 0)
    shrink = cfg_model["shrink"]
    reg = pa[(pa.get("game_type", "R") == "R") & pa["outcome"].isin(OUTCOMES)].copy()
    reg["stand"] = reg["stand"].astype(str)
    reg["p_throws"] = reg["p_throws"].astype(str)
    train, test = reg[reg["game_date"] < split], reg[reg["game_date"] >= split]
    if len(train) < 1000 or len(test) < 500:
        raise ValueError(f"Not enough data to backtest (train {len(train)}, test {len(test)}).")
    y = test["outcome"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()

    # choose lambda on a validation window carved off the end of training
    val_start = split - pd.Timedelta(days=30)
    fit_pa, val_pa = train[train["game_date"] < val_start], train[train["game_date"] >= val_start]
    fit_pitch = pitches[pitches["game_date"] < val_start]
    # tune regression strength and recency on the validation window
    tune = {}
    for mult in (0.5, 1.0, 2.0, 3.0, 4.0):
        for h in (0, 60, 120):
            k_pa = {o: v * mult for o, v in shrink["pa"].items()}
            P = _predict(build_rates(fit_pa, k_pa, h, val_start), val_pa, None, 0.0)
            tune[(mult, h)] = _logloss(P, val_pa["outcome"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy())
    best_mult, hl = min(tune, key=tune.get)
    shrink_pa = {o: v * best_mult for o, v in shrink["pa"].items()}
    if verbose:
        print(f"  tuned rates: shrink x{best_mult}, recency half-life {hl} days")
    r_fit = build_rates(fit_pa, shrink_pa, hl, val_start)
    prof_fit = build_profiles(fit_pitch, shrink, hl, val_start)
    e_val = pair_edges(prof_fit, val_pa["batter"].to_numpy(), val_pa["pitcher"].to_numpy(),
                       val_pa["stand"].to_numpy())
    y_val = val_pa["outcome"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    base_val = _predict(r_fit, val_pa, None, 0.0)
    grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15]
    val_scores = {}
    for lam in grid:
        P = base_val.copy()
        if lam:
            tilt = np.where(GOOD[None, :], np.exp(lam * e_val[:, None]), np.exp(-lam * e_val[:, None]))
            P = P * tilt
            P = P / P.sum(1, keepdims=True)
        val_scores[lam] = _logloss(P, y_val)
    lam_best = min(val_scores, key=val_scores.get)

    # pitch quality: choose the weight of the stuff tilt on the same validation window
    from .stuff import fit_stuff, stability_check
    sm_fit = fit_stuff(fit_pitch[fit_pitch["game_type"].astype(str).eq("R")], verbose=verbose)
    s_val = val_pa["pitcher"].map(sm_fit.ratings["rv100"]).fillna(0.0).to_numpy()
    lam_tilt_val = np.where(GOOD[None, :], np.exp(lam_best * e_val[:, None]), np.exp(-lam_best * e_val[:, None]))
    base_mix_val = base_val * lam_tilt_val
    base_mix_val = base_mix_val / base_mix_val.sum(1, keepdims=True)
    g_scores = {}
    for g in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        P = base_mix_val * np.where(GOOD[None, :], np.exp(-g * s_val[:, None]), np.exp(g * s_val[:, None]))
        g_scores[g] = _logloss(P / P.sum(1, keepdims=True), y_val)
    gamma_best = min(g_scores, key=g_scores.get)

    # refit on all training data and score the test window
    r = build_rates(train, shrink_pa, hl, split)
    prof = build_profiles(pitches[pitches["game_date"] < split], shrink, hl, split)
    edges = pair_edges(prof, test["batter"].to_numpy(), test["pitcher"].to_numpy(), test["stand"].to_numpy())
    P_or = _predict(r, test, None, 0.0)
    tilt = np.where(GOOD[None, :], np.exp(lam_best * edges[:, None]), np.exp(-lam_best * edges[:, None]))
    P_mix = P_or * tilt
    P_mix = P_mix / P_mix.sum(1, keepdims=True)
    P_lg = np.vstack([r.league.get((s, h), np.mean(list(r.league.values()), axis=0))
                      for s, h in zip(test["stand"], test["p_throws"])])
    sm = fit_stuff(pitches[(pitches["game_date"] < split) & pitches["game_type"].astype(str).eq("R")], verbose=verbose)
    s_test = test["pitcher"].map(sm.ratings["rv100"]).fillna(0.0).to_numpy()
    P_stuff = P_mix * np.where(GOOD[None, :], np.exp(-gamma_best * s_test[:, None]), np.exp(gamma_best * s_test[:, None]))
    P_stuff = P_stuff / P_stuff.sum(1, keepdims=True)
    reg_pitch = pitches[pitches["game_type"].astype(str).eq("R")]
    stab = stability_check(reg_pitch, split)

    w = np.array([WOBA_W[o] for o in OUTCOMES])
    act = w[y]
    exp_or = P_or @ w
    resid = act - exp_or
    # does the arsenal edge predict what the talent model misses?
    corr = float(np.corrcoef(edges, resid)[0, 1]) if np.std(edges) > 0 else 0.0
    q = pd.qcut(edges, 5, labels=False, duplicates="drop")
    by_q = pd.DataFrame({"q": q, "edge": edges, "resid": resid}).groupby("q").agg(
        edge=("edge", "mean"), resid=("resid", "mean"), n=("resid", "size"))
    # calibration of the talent model
    cal = pd.DataFrame({"exp": exp_or, "act": act})
    cal["bin"] = pd.qcut(cal["exp"], 10, labels=False, duplicates="drop")
    cal = cal.groupby("bin").agg(exp=("exp", "mean"), act=("act", "mean"), n=("act", "size"))

    res = {
        "train_pa": int(len(train)), "test_pa": int(len(test)), "split": str(split.date()),
        "logloss": {"league": _logloss(P_lg, y), "oddsratio": _logloss(P_or, y), "oddsratio_mix": _logloss(P_mix, y),
                    "with_stuff": _logloss(P_stuff, y)},
        "lambda": lam_best, "lambda_validation": {str(k): v for k, v in val_scores.items()},
        "tuned": {"shrink_mult": best_mult, "half_life": hl,
                  "grid": [{"shrink_mult": m_, "half_life": h_, "logloss": v} for (m_, h_), v in tune.items()]},
        "edge_resid_corr": corr,
        "edge_quintiles": by_q.reset_index().round(4).to_dict(orient="records"),
        "calibration": cal.reset_index().round(4).to_dict(orient="records"),
    }
    ll = res["logloss"]
    res["skill_vs_league_pct"] = 100 * (ll["league"] - ll["oddsratio"]) / ll["league"]
    res["mix_gain_pct"] = 100 * (ll["oddsratio"] - ll["oddsratio_mix"]) / ll["oddsratio"]
    res["stuff"] = {"gamma": gamma_best, "validation": {str(k): v for k, v in g_scores.items()},
                    "gain_pct": 100 * (ll["oddsratio_mix"] - ll["with_stuff"]) / ll["oddsratio_mix"],
                    "stability": stab, "rating_sd": sm.sd}
    if verbose:
        print(f"Backtest: train {res['train_pa']:,} PA before {res['split']}, test {res['test_pa']:,} PA")
        for kname, v in ll.items():
            print(f"  log loss {kname:<14} {v:.5f}")
        print(f"  talent model beats league-only by {res['skill_vs_league_pct']:.2f}%")
        print(f"  pitch-mix tilt (lambda={lam_best}) changes log loss by {res['mix_gain_pct']:+.3f}%")
        print(f"  corr(edge, wOBA residual) = {corr:+.4f}")
        print(f"  stuff tilt (gamma={gamma_best}) changes log loss by {res['stuff']['gain_pct']:+.3f}%; stability {stab}")
        import os
        if os.environ.get("GITHUB_ACTIONS"):
            print("::notice title=backtest::" + "; ".join(f"{k} {v:.5f}" for k, v in ll.items())
                  + f"; lambda {lam_best}; mix gain {res['mix_gain_pct']:+.3f}%; corr {corr:+.4f}", flush=True)
            st = res["stuff"]
            print(f"::notice title=stuff::gamma {gamma_best}; test gain {st['gain_pct']:+.3f}%; validation "
                  + ", ".join(f"{k}:{v:.5f}" for k, v in st["validation"].items())
                  + f"; stability {stab}", flush=True)
    return res
