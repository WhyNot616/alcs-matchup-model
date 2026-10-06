"""Pitch-quality ("stuff") model.

Predicts the count-neutral expected run value of a pitch from its physical characteristics alone:
velocity, movement, spin, extension, release point and arm angle, plus how it differs from the same
pitcher's primary fastball. Location and results are deliberately left out, so the model rates what a
pitch does, not where it went or what happened to it. This is the same idea behind public Stuff+
models.

Why it can help: a pitcher's results take hundreds of plate appearances to stabilize, while his stuff
is measurable within a few hundred pitches. A pitcher whose stuff is better than his results is
likely to pitch better going forward, and the market may be slow to see it.

Outputs, per pitcher: stuff runs saved per 100 pitches (pitcher view, regressed) and Stuff+ (100 =
league average, 10 = one standard deviation across pitchers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FB_TYPES = ("FF", "SI", "FC")
FEATURES = ["velo", "pfx_x_n", "pfx_z", "spin", "ext", "rel_x_n", "rel_z", "arm_angle", "axis_n",
            "grp", "d_velo", "d_pfx_x", "d_pfx_z", "same_hand"]
GRP_CODE = {"FB": 0, "BR": 1, "OS": 2}


@dataclass
class StuffModel:
    model: object
    count_mean: dict
    fb_ref: pd.DataFrame          # pitcher -> primary fastball velo / movement
    ratings: pd.DataFrame         # pitcher -> n, rv100 (pitcher view, regressed), stuff_plus
    sd: float
    shrink: float
    cols: list = None             # features actually used (all-missing columns are dropped)


def _num(d: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)


def fastball_reference(p: pd.DataFrame) -> pd.DataFrame:
    """Each pitcher's most-used fastball type and its average velocity and movement."""
    fb = p[p["pitch_type"].isin(FB_TYPES)]
    if fb.empty:
        return pd.DataFrame(columns=["fb_velo", "fb_pfx_x_n", "fb_pfx_z"])
    lefty = fb["p_throws"].astype(str).eq("L")
    fb = fb.assign(pfx_x_n=np.where(lefty, -_num(fb, "pfx_x"), _num(fb, "pfx_x")), pfx_z_=_num(fb, "pfx_z"),
                   velo_=_num(fb, "release_speed"))
    top = fb.groupby(["pitcher", "pitch_type"], observed=True).size().reset_index(name="n")
    top = top.sort_values("n", ascending=False).drop_duplicates("pitcher")
    m = fb.merge(top[["pitcher", "pitch_type"]], on=["pitcher", "pitch_type"])
    return m.groupby("pitcher").agg(fb_velo=("velo_", "mean"), fb_pfx_x_n=("pfx_x_n", "mean"),
                                    fb_pfx_z=("pfx_z_", "mean"))


def features(p: pd.DataFrame, fb_ref: pd.DataFrame) -> pd.DataFrame:
    lefty = p["p_throws"].astype(str).eq("L").to_numpy()
    sgn = np.where(lefty, -1.0, 1.0)
    X = pd.DataFrame(index=p.index)
    X["velo"] = _num(p, "release_speed")
    X["pfx_x_n"] = _num(p, "pfx_x") * sgn
    X["pfx_z"] = _num(p, "pfx_z")
    X["spin"] = _num(p, "release_spin_rate")
    X["ext"] = _num(p, "release_extension")
    X["rel_x_n"] = _num(p, "release_pos_x") * sgn
    X["rel_z"] = _num(p, "release_pos_z")
    X["arm_angle"] = _num(p, "arm_angle")
    axis = _num(p, "spin_axis")
    X["axis_n"] = np.where(lefty, 360 - axis, axis)
    X["grp"] = p["pgroup"].map(GRP_CODE).astype(float)
    ref = fb_ref.reindex(p["pitcher"].to_numpy())
    X["d_velo"] = X["velo"].to_numpy() - ref["fb_velo"].to_numpy()
    X["d_pfx_x"] = X["pfx_x_n"].to_numpy() - ref["fb_pfx_x_n"].to_numpy()
    X["d_pfx_z"] = X["pfx_z"].to_numpy() - ref["fb_pfx_z"].to_numpy()
    X["same_hand"] = (p["stand"].astype(str).to_numpy() == p["p_throws"].astype(str).to_numpy()).astype(float)
    return X[FEATURES]


def _target(p: pd.DataFrame, count_mean: dict | None = None):
    key = (p["balls"].fillna(0).astype(int) * 10 + p["strikes"].fillna(0).astype(int)).to_numpy()
    xrv = p["xrv"].astype(float).to_numpy()
    if count_mean is None:
        count_mean = pd.Series(xrv).groupby(key).mean().to_dict()
    return xrv - pd.Series(key).map(count_mean).fillna(0.0).to_numpy(), count_mean


def fit_stuff(p: pd.DataFrame, shrink: float = 150, max_rows: int = 500_000, seed: int = 0,
              verbose: bool = True) -> StuffModel:
    """Train on pitches in `p` and rate every pitcher in it."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    d = p[p["pgroup"].notna() & p["release_speed"].notna()]
    fb_ref = fastball_reference(d)
    y, cm = _target(d)
    X = features(d, fb_ref)
    cols = [c for c in FEATURES if X[c].notna().mean() > 0.05 and X[c].nunique() > 1]
    X = X[cols]
    if len(d) > max_rows:
        idx = np.random.default_rng(seed).choice(len(d), max_rows, replace=False)
        Xf, yf = X.iloc[idx], y[idx]
    else:
        Xf, yf = X, y
    model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=400, l2_regularization=1.0,
                                          categorical_features=[cols.index("grp")] if "grp" in cols else None,
                                          random_state=seed)
    model.fit(Xf, yf)
    sm = StuffModel(model, cm, fb_ref, pd.DataFrame(), 1.0, shrink, cols)
    sm.ratings, sm.sd = rate_pitchers(sm, d)
    if verbose:
        r = sm.ratings
        print(f"  stuff model: {len(Xf):,} pitches, {int((r['n'] >= 300).sum())} pitchers with 300+ pitches, "
              f"SD across pitchers {sm.sd:.3f} runs/100", flush=True)
    return sm


def predict_pitch(sm: StuffModel, p: pd.DataFrame) -> np.ndarray:
    d = p[p["pgroup"].notna() & p["release_speed"].notna()]
    out = np.full(len(p), np.nan)
    if len(d):
        pos = p.index.get_indexer(d.index)
        out[pos] = sm.model.predict(features(d, sm.fb_ref)[sm.cols or FEATURES])
    return out


def rate_pitchers(sm: StuffModel, p: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Average predicted run value per pitcher, pitcher view, regressed toward average."""
    pred = predict_pitch(sm, p)
    d = pd.DataFrame({"pitcher": p["pitcher"].to_numpy(), "v": -pred * 100}).dropna()
    g = d.groupby("pitcher")["v"].agg(["mean", "size"]).rename(columns={"mean": "raw", "size": "n"})
    lg = float(np.average(g["raw"], weights=g["n"]))
    g["rv100"] = (g["raw"] - lg) * g["n"] / (g["n"] + sm.shrink)
    big = g[g["n"] >= 500]
    sd = float(np.sqrt(np.average((big["raw"] - lg) ** 2, weights=big["n"]))) if len(big) > 5 else float(g["raw"].std() or 1.0)
    g["stuff_plus"] = 100 + 10 * g["rv100"] / (sd or 1.0)
    return g[["n", "rv100", "stuff_plus"]], sd


def stability_check(p: pd.DataFrame, split: pd.Timestamp, min_pitches: int = 400) -> dict:
    """Does early-season stuff predict late-season results better than early-season results do?

    Fits the stuff model on pitches before `split`, rates pitchers on that window, and correlates the
    ratings with each pitcher's results (expected run value per 100, pitcher view) after `split`.
    """
    a, b = p[p["game_date"] < split], p[p["game_date"] >= split]
    sm = fit_stuff(a, verbose=False)
    res_a = (-a.groupby("pitcher")["xrv"].mean() * 100).rename("results_before")
    res_b = (-b.groupby("pitcher")["xrv"].mean() * 100).rename("results_after")
    n_a, n_b = a.groupby("pitcher").size(), b.groupby("pitcher").size()
    df = pd.concat([sm.ratings["rv100"].rename("stuff_before"), res_a, res_b], axis=1)
    keep = (n_a.reindex(df.index) >= min_pitches) & (n_b.reindex(df.index) >= min_pitches)
    df = df[keep.fillna(False)].dropna()
    if len(df) < 20:
        return {"n": int(len(df))}
    c = df.corr()
    both = np.linalg.lstsq(np.column_stack([np.ones(len(df)), df["stuff_before"], df["results_before"]]),
                           df["results_after"].to_numpy(), rcond=None)[0]
    pred = np.column_stack([np.ones(len(df)), df["stuff_before"], df["results_before"]]) @ both
    r_both = float(np.corrcoef(pred, df["results_after"])[0, 1])
    return {"n": int(len(df)), "min_pitches": min_pitches, "split": str(split.date()),
            "r_stuff": float(c.loc["stuff_before", "results_after"]),
            "r_results": float(c.loc["results_before", "results_after"]),
            "r_both": r_both}
