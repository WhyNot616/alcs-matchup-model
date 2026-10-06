"""Bullpen tendencies, starter hooks and a usage model learned from each team's 2026 games.

Appearance table: one row per pitcher per game, with the state when he entered (inning, outs, bases,
score margin from the fielding team's view, leverage), how long he stayed, rest, and results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import add_leverage, leverage_table, plate_appearances

LEAD_BUCKETS = [(-99, -4, "down 4+"), (-3, -1, "down 1-3"), (0, 0, "tied"), (1, 3, "up 1-3"), (4, 99, "up 4+")]


def lead_bucket(margin: float) -> str:
    for lo, hi, name in LEAD_BUCKETS:
        if lo <= margin <= hi:
            return name
    return "tied"


def appearances(pitches: pd.DataFrame, team: str, li_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every pitching appearance for `team`, with entry state and outcomes."""
    d = pitches[pitches["fld_team"].eq(team)].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    pa = plate_appearances(d)
    if li_table is None:
        li_table = leverage_table(plate_appearances(pitches))
    pa = add_leverage(pa, li_table)
    first_p = d.groupby("game_pk", sort=False)["pitcher"].first().rename("starter")
    g = d.groupby(["game_pk", "pitcher"], sort=False)
    ent = g.head(1).set_index(["game_pk", "pitcher"])
    app = pd.DataFrame({
        "game_date": ent["game_date"], "game_type": ent.get("game_type"),
        "entry_inning": ent["inning"], "entry_outs": ent["outs_when_up"], "entry_bases": ent["base_state"],
        "entry_margin": -ent["score_diff_bat"],  # fielding team's lead
        "days_rest": ent["pitcher_days_since_prev_game"], "first_stand": ent["stand"], "throws": ent["p_throws"],
    })
    app["pitches"] = g.size()
    app["fb_velo"] = d[d["pgroup"].eq("FB")].groupby(["game_pk", "pitcher"])["release_speed"].mean()
    pa = pa.assign(_k=pa["outcome"].eq("K"), _bb=pa["outcome"].eq("BB"),
                   _out=pa["outcome"].isin(["K", "OUT"]),
                   _xw=pa["xwoba"].fillna(0) * pa["woba_denom"].fillna(0), _den=pa["woba_denom"].fillna(0))
    pa_g = pa.groupby(["game_pk", "pitcher"])
    app["bf"] = pa_g.size()
    app["entry_li"] = pa.groupby(["game_pk", "pitcher"]).head(1).set_index(["game_pk", "pitcher"])["li"]
    app["avg_li"] = pa_g["li"].mean()
    app["k"] = pa_g["_k"].sum()
    app["bb"] = pa_g["_bb"].sum()
    app["xw_sum"] = pa_g["_xw"].sum()
    app["w_den"] = pa_g["_den"].sum()
    # outs recorded: PA-ending outs (approximate; double plays count once)
    app["outs"] = pa_g["_out"].sum()
    app[["bf", "k", "bb", "xw_sum", "w_den", "outs"]] = app[["bf", "k", "bb", "xw_sum", "w_den", "outs"]].fillna(0)
    app = app.reset_index()
    app["is_start"] = app["pitcher"].to_numpy() == first_p.reindex(app["game_pk"]).to_numpy()
    app["inherited"] = app["entry_bases"].map(lambda b: bin(int(b)).count("1") if pd.notna(b) else 0)
    app["lead_bucket"] = app["entry_margin"].map(lead_bucket)
    app["save_sit"] = (app["entry_inning"] >= 9) & app["entry_margin"].between(1, 3)
    app["high_lev"] = app["entry_li"] >= 1.5
    app["team"] = team
    return app


def reliever_tendencies(app: pd.DataFrame, names: dict[int, str], roster: list[int] | None = None) -> pd.DataFrame:
    r = app[~app["is_start"]]
    if roster is not None:
        r = r[r["pitcher"].isin(roster)]
    g = r.groupby("pitcher")
    out = pd.DataFrame({
        "g": g.size(),
        "gmLI": g["entry_li"].mean(),
        "high_lev_pct": g["high_lev"].mean(),
        "save_sit_pct": g["save_sit"].mean(),
        "inn9_pct": g["entry_inning"].apply(lambda s: (s >= 9).mean()),
        "avg_entry_inning": g["entry_inning"].mean(),
        "multi_inning_pct": g["outs"].apply(lambda s: (s >= 4).mean()),
        "avg_pitches": g["pitches"].mean(),
        "inherited_pct": g["inherited"].apply(lambda s: (s > 0).mean()),
        "b2b_apps": g["days_rest"].apply(lambda s: (s == 1).sum()),
        "k_pct": g["k"].sum() / g["bf"].sum(),
        "bb_pct": g["bb"].sum() / g["bf"].sum(),
        "xwoba": g["xw_sum"].sum() / g["w_den"].sum(),
    })
    # same-hand first batter (a manager's matchup tendency)
    sh = r.assign(same=r["first_stand"].astype(str) == r["throws"].astype(str)).groupby("pitcher")["same"].mean()
    out["same_hand_first_pct"] = sh
    # fatigue: fastball velo on back-to-back days vs rested
    rest = r.assign(b2b=r["days_rest"] == 1).groupby(["pitcher", "b2b"])["fb_velo"].mean().unstack()
    if True in rest.columns and False in rest.columns:
        out["b2b_velo_drop"] = rest[True] - rest[False]
    else:
        out["b2b_velo_drop"] = np.nan
    hl = r[r["high_lev"]].groupby("pitcher")
    out["hl_xwoba"] = hl["xw_sum"].sum() / hl["w_den"].sum()
    out["name"] = [names.get(int(p), str(p)) for p in out.index]
    out["role"] = out.apply(_role, axis=1)
    return out.sort_values("gmLI", ascending=False)


def _role(row) -> str:
    if row["g"] >= 15 and row["save_sit_pct"] >= 0.35:
        return "closer"
    if row["gmLI"] >= 1.3 and row["g"] >= 15:
        return "high leverage"
    if row["multi_inning_pct"] >= 0.4:
        return "long / bulk"
    if row["gmLI"] >= 0.9:
        return "middle"
    return "low leverage"


def situational_splits(pitches: pd.DataFrame, team: str, roster: list[int], names: dict[int, str]) -> pd.DataFrame:
    """Reliever results by situation: RISP, two outs, late & close, first batter faced, two strikes."""
    d = pitches[pitches["fld_team"].eq(team) & pitches["pitcher"].isin(roster)]
    pa = plate_appearances(d)
    pa["risp"] = pa["base_state"].isin([2, 3, 4, 5, 6, 7])
    pa["two_out"] = pa["outs_when_up"].eq(2)
    pa["late_close"] = (pa["inning"] >= 7) & (pa["score_diff_bat"].abs() <= 2)
    pa["first_bf"] = ~pa.duplicated(["game_pk", "pitcher"])
    rows = []
    for pid, g in pa.groupby("pitcher"):
        row = {"pitcher": int(pid), "name": names.get(int(pid), str(pid))}
        for col in ("risp", "two_out", "late_close", "first_bf"):
            s = g[g[col]]
            den = s["woba_denom"].fillna(0).sum()
            row[f"{col}_pa"] = int(len(s))
            row[f"{col}_xwoba"] = float((s["xwoba"] * s["woba_denom"].fillna(0)).sum() / den) if den else None
            row[f"{col}_k"] = float((s["outcome"] == "K").mean()) if len(s) else None
        rows.append(row)
    return pd.DataFrame(rows)


def starter_hooks(app: pd.DataFrame) -> pd.DataFrame:
    """How long each pitcher lasts when he starts: batters faced and pitch count distributions."""
    s = app[app["is_start"]]
    g = s.groupby("pitcher")
    return pd.DataFrame({
        "gs": g.size(), "bf_mean": g["bf"].mean(), "bf_p25": g["bf"].quantile(.25), "bf_p75": g["bf"].quantile(.75),
        "pitches_mean": g["pitches"].mean(), "pitches_max": g["pitches"].max(),
        "through_3x_pct": g["bf"].apply(lambda x: (x >= 19).mean()),
    })


class UsageModel:
    """Chooses relievers the way a manager did in 2026.

    For each (inning bucket, lead bucket) state, P(reliever) = smoothed share of that state's
    appearances, blended with the reliever's overall share. In October the top relievers by gmLI
    get a multiplier on high-leverage states (postseason_leverage_boost).
    """

    def __init__(self, app: pd.DataFrame, roster: list[int], boost: float = 1.0, alpha: float = 3.0):
        r = app[(~app["is_start"]) & app["pitcher"].isin(roster)].copy()
        r["inn_b"] = r["entry_inning"].clip(lower=5, upper=10).astype(int)
        self.roster = [p for p in roster if p in set(r["pitcher"])] or list(roster)
        tot = r.groupby("pitcher").size().reindex(self.roster).fillna(0) + 0.5
        self.base = (tot / tot.sum()).to_dict()
        self.gmli = r.groupby("pitcher")["entry_li"].mean().reindex(self.roster).fillna(1.0).to_dict()
        top = sorted(self.roster, key=lambda p: -self.gmli[p])[: max(2, len(self.roster) // 3)]
        self.top = set(top)
        self.boost = boost
        self.table: dict[tuple[int, str], dict[int, float]] = {}
        for (ib, lb), g in r.groupby(["inn_b", "lead_bucket"]):
            c = g.groupby("pitcher").size()
            n = c.sum()
            self.table[(int(ib), lb)] = {p: (c.get(p, 0) + alpha * self.base[p]) / (n + alpha) for p in self.roster}
        # outs-recorded distribution per reliever (for outing length)
        self.outs_dist = {}
        for p, g in r.groupby("pitcher"):
            v = g["outs"].clip(lower=1, upper=9).astype(int).value_counts(normalize=True).sort_index()
            self.outs_dist[int(p)] = (v.index.to_numpy(), v.to_numpy())

    def choose(self, inning: int, margin: int, available: list[int], rng: np.random.Generator) -> int | None:
        cand = [p for p in available if p in self.base]
        if not cand:
            return available[0] if available else None
        key = (int(min(max(inning, 5), 10)), lead_bucket(margin))
        probs = self.table.get(key, self.base)
        high = abs(margin) <= 2 and inning >= 7
        w = np.array([probs.get(p, self.base[p]) * (self.boost if (high and p in self.top) else 1.0) for p in cand])
        w = w / w.sum()
        return int(rng.choice(cand, p=w))

    def outing_outs(self, pitcher: int, rng: np.random.Generator) -> int:
        if pitcher in self.outs_dist:
            vals, p = self.outs_dist[pitcher]
            return int(rng.choice(vals, p=p))
        return 3
