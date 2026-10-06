"""Pitch-type run-value profiles with hierarchical shrinkage, and hitter-vs-pitcher arsenal edges.

Everything here uses expected run value (xrv, see features.expected_run_value) per 100 pitches.
Values are expressed relative to the league average for that pitch type, so +1.0 means one run per
100 pitches better than an average hitter (or pitcher) against (or with) that pitch.

Shrinkage is two-level:
  pitch group (FB / BR / OS)  -> regressed toward the league average for the group
  pitch type (FF, SL, ...)    -> regressed toward the player's own group estimate
so a hitter with 40 splitters is pulled toward how he handles all offspeed pitches, not toward zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import recency_weights


@dataclass
class Profiles:
    hitter: pd.DataFrame          # index (batter, pitch_type): n, raw, rel, whiff
    pitcher: pd.DataFrame         # index (pitcher, pitch_type): n, raw, rel, whiff, velo
    usage: pd.DataFrame           # index (pitcher, stand, pitch_type): u
    league: pd.DataFrame          # index pitch_type: rv100 (hitter view), group
    hitter_group: pd.DataFrame    # index (batter, pgroup): n, rel


def _wsum(df: pd.DataFrame, keys: list[str], w: str = "w") -> pd.DataFrame:
    g = df.assign(_wx=df["xrv"] * df[w], _ww=df["is_whiff"] * df[w], _ws=df["is_swing"] * df[w])
    out = g.groupby(keys, observed=True).agg(n=(w, "sum"), xs=("_wx", "sum"), wh=("_ww", "sum"),
                                             sw=("_ws", "sum"), cnt=(w, "size"))
    out["raw"] = out["xs"] / out["n"] * 100
    out["whiff"] = out["wh"] / out["sw"].replace(0, np.nan)
    return out


def build_profiles(pitches: pd.DataFrame, shrink: dict, half_life: float = 0,
                   as_of: pd.Timestamp | None = None) -> Profiles:
    d = pitches[pitches["pgroup"].notna()].copy()
    d["w"] = recency_weights(d["game_date"], half_life, as_of)
    if "bw" in d.columns:
        d["w"] = d["w"] * d["bw"].to_numpy()
    k_grp, k_pt, k_pit = shrink["pitch_group"], shrink["pitch_type"], shrink["pitcher_pitch"]

    lg_pt = _wsum(d, ["pitch_type"])
    lg_grp = _wsum(d, ["pgroup"])
    league = pd.DataFrame({"rv100": lg_pt["raw"]})
    pt_group = d.drop_duplicates("pitch_type").set_index("pitch_type")["pgroup"]
    league["group"] = pt_group.reindex(league.index)
    lg_pt_raw = league["rv100"]
    lg_grp_raw = lg_grp["raw"]

    # ---- hitters (hitter view)
    hg = _wsum(d, ["batter", "pgroup"])
    lg_g = lg_grp_raw.reindex(hg.index.get_level_values("pgroup")).to_numpy()
    hg["shr"] = (hg["raw"] * hg["n"] + lg_g * k_grp) / (hg["n"] + k_grp)
    hg["rel"] = hg["shr"] - lg_g
    ht = _wsum(d, ["batter", "pitch_type"])
    b = ht.index.get_level_values("batter")
    pt = ht.index.get_level_values("pitch_type")
    grp = pt_group.reindex(pt).to_numpy()
    grp_rel = hg["rel"].reindex(pd.MultiIndex.from_arrays([b, grp])).fillna(0).to_numpy()
    prior = lg_pt_raw.reindex(pt).to_numpy() + grp_rel
    ht["shr"] = (ht["raw"] * ht["n"] + prior * k_pt) / (ht["n"] + k_pt)
    ht["rel"] = ht["shr"] - lg_pt_raw.reindex(pt).to_numpy()

    # ---- pitchers (pitcher view: positive = runs saved)
    pg = _wsum(d, ["pitcher", "pgroup"])
    lg_gp = -lg_grp_raw.reindex(pg.index.get_level_values("pgroup")).to_numpy()
    pg["shr"] = (-pg["raw"] * pg["n"] + lg_gp * k_grp) / (pg["n"] + k_grp)
    pg["rel"] = pg["shr"] - lg_gp
    ptab = _wsum(d, ["pitcher", "pitch_type"])
    p = ptab.index.get_level_values("pitcher")
    ppt = ptab.index.get_level_values("pitch_type")
    pgrp = pt_group.reindex(ppt).to_numpy()
    pgrp_rel = pg["rel"].reindex(pd.MultiIndex.from_arrays([p, pgrp])).fillna(0).to_numpy()
    lg_p = -lg_pt_raw.reindex(ppt).to_numpy()
    pprior = lg_p + pgrp_rel
    ptab["shr"] = (-ptab["raw"] * ptab["n"] + pprior * k_pit) / (ptab["n"] + k_pit)
    ptab["rel"] = ptab["shr"] - lg_p
    ptab["raw_p"] = -ptab["raw"]
    velo = d.groupby(["pitcher", "pitch_type"], observed=True)["release_speed"].mean()
    ptab["velo"] = velo.reindex(ptab.index)

    # ---- usage by batter side, shrunk toward the pitcher's overall mix (20 pseudo-pitches)
    tot_all = d.groupby(["pitcher", "pitch_type"], observed=True)["w"].sum()
    u_all = tot_all / tot_all.groupby(level=0).transform("sum")
    cnt = d.groupby(["pitcher", "stand", "pitch_type"], observed=True)["w"].sum().rename("c").reset_index()
    nside = d.groupby(["pitcher", "stand"], observed=True)["w"].sum().rename("n_side").reset_index()
    grid = u_all.rename("u_all").reset_index().merge(pd.DataFrame({"stand": ["L", "R"]}), how="cross")
    grid["stand"] = grid["stand"].astype(str)
    cnt["stand"] = cnt["stand"].astype(str)
    nside["stand"] = nside["stand"].astype(str)
    grid = grid.merge(cnt, on=["pitcher", "stand", "pitch_type"], how="left")
    grid = grid.merge(nside, on=["pitcher", "stand"], how="left")
    grid[["c", "n_side"]] = grid[["c", "n_side"]].fillna(0.0)
    grid["u"] = (grid["c"] + 20 * grid["u_all"]) / (grid["n_side"] + 20)
    usage = grid
    usage["u"] = usage["u"] / usage.groupby(["pitcher", "stand"])["u"].transform("sum")
    usage = usage.set_index(["pitcher", "stand", "pitch_type"])[["u"]].sort_index()

    return Profiles(
        hitter=ht[["n", "cnt", "raw", "rel", "whiff"]],
        pitcher=ptab[["n", "cnt", "raw_p", "rel", "whiff", "velo"]].rename(columns={"raw_p": "raw"}),
        usage=usage,
        league=league,
        hitter_group=hg[["n", "rel"]],
    )


def pair_edges(prof: Profiles, batters: np.ndarray, pitchers: np.ndarray, stands: np.ndarray) -> np.ndarray:
    """Arsenal edge (hitter view, runs per 100 pitches) for many PAs at once.

    edge = sum over the pitcher's pitch types of usage_vs_side * (hitter_rel - pitcher_rel)
    Hitters with no history on a pitch fall back to their pitch-group estimate (or 0).
    """
    use = prof.usage.reset_index()
    hrel = prof.hitter["rel"]
    prel = prof.pitcher["rel"]
    pairs = pd.DataFrame({"batter": batters.astype("int64"), "pitcher": pitchers.astype("int64"),
                          "stand": stands.astype(str)}).reset_index(names="row")
    m = pairs.merge(use, on=["pitcher", "stand"], how="left")
    m = m[m["u"].notna()]
    m["h"] = hrel.reindex(pd.MultiIndex.from_arrays([m["batter"], m["pitch_type"]])).to_numpy()
    grp = prof.league["group"].reindex(m["pitch_type"]).to_numpy()
    hg = prof.hitter_group["rel"].reindex(pd.MultiIndex.from_arrays([m["batter"], grp])).to_numpy()
    m["h"] = np.where(np.isnan(m["h"]), np.nan_to_num(hg), m["h"])
    m["p"] = prel.reindex(pd.MultiIndex.from_arrays([m["pitcher"], m["pitch_type"]])).fillna(0).to_numpy()
    m["c"] = m["u"] * (m["h"] - m["p"])
    out = np.zeros(len(pairs))
    s = m.groupby("row")["c"].sum()
    out[s.index.to_numpy()] = s.to_numpy()
    return out


def matchup_detail(prof: Profiles, batter: int, pitcher: int, stand: str) -> list[dict]:
    """Per-pitch breakdown for one hitter vs one pitcher (used by the dashboard)."""
    try:
        u = prof.usage.loc[(pitcher, stand)]["u"]
    except KeyError:
        return []
    rows = []
    for pt, uu in u.sort_values(ascending=False).items():
        h = prof.hitter["rel"].get((batter, pt), np.nan)
        hn = prof.hitter["cnt"].get((batter, pt), 0)
        if np.isnan(h):
            grp = prof.league["group"].get(pt)
            h = prof.hitter_group["rel"].get((batter, grp), 0.0)
        p = prof.pitcher["rel"].get((pitcher, pt), 0.0)
        rows.append({"pt": pt, "u": round(float(uu), 4), "h": round(float(h), 3), "hn": int(hn),
                     "p": round(float(p), 3), "edge": round(float(uu * (h - p)), 3),
                     "velo": None if pd.isna(prof.pitcher["velo"].get((pitcher, pt), np.nan))
                     else round(float(prof.pitcher["velo"].get((pitcher, pt))), 1),
                     "hw": _r(prof.hitter["whiff"].get((batter, pt), np.nan)),
                     "pw": _r(prof.pitcher["whiff"].get((pitcher, pt), np.nan))})
    return rows


def _r(v, d=3):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), d)
