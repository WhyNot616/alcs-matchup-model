"""End-to-end pipeline: load data, fit the models, simulate, and write outputs for the dashboard."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from . import aggregates as agg
from .bullpen import UsageModel, appearances, reliever_tendencies, situational_splits, starter_hooks
from .config import DASHBOARD_TEMPLATE, DOCS, OUTPUT, Config, ensure_dirs
from .data import load_leaderboard, load_schedule, load_statcast, lookup_names
from .features import (OUTCOMES, active_players, leverage_table, plate_appearances, player_names,
                       prepare_pitches)
from .matchups import build_profiles, matchup_detail, pair_edges
from .pa_model import backtest, batter_side, build_rates, expected_woba, matchup_probs
from .simulate import GamePlan, SimContext, TeamSetup, calibrate, calibrate_scoring, sim_series


@dataclass
class Fitted:
    pitches: pd.DataFrame
    pa: pd.DataFrame
    rates: object
    profiles: object
    li: pd.DataFrame
    as_of: pd.Timestamp
    stuff: pd.DataFrame | None = None     # pitcher -> n, rv100, stuff_plus


def _log(msg: str, t0: float) -> None:
    line = f"[{time.time() - t0:6.1f}s] {msg}"
    print(line, flush=True)
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::notice title=build::{line}", flush=True)


def load_prepared(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_statcast(cfg)
    p = prepare_pitches(raw)
    pa = plate_appearances(p)
    return p, pa


def fit(cfg: Config, pitches: pd.DataFrame, pa: pd.DataFrame, tuned: dict | None = None,
        stuff_gamma: float = 0.0, stuff_ratings: pd.DataFrame | None = None) -> Fitted:
    """Fit outcome rates and pitch profiles. `tuned` (from the backtest) overrides shrinkage and recency.
    With stuff_gamma > 0 the stuff model is fit too (or `stuff_ratings` reused) and feeds the rates."""
    m = cfg.model
    as_of = pitches["game_date"].max()
    hl = m.get("recent_half_life_days", 0)
    k_pa = dict(m["shrink"]["pa"])
    if tuned:
        hl = tuned.get("half_life", hl)
        k_pa = {o: v * tuned.get("shrink_mult", 1.0) for o, v in k_pa.items()}
    rates = build_rates(pa, k_pa, hl, as_of)
    prof = build_profiles(pitches, m["shrink"], hl, as_of)
    li = leverage_table(pa)
    F = Fitted(pitches, pa, rates, prof, li, as_of)
    if stuff_ratings is None and (stuff_gamma or cfg.model.get("always_rate_stuff", True)):
        from .stuff import fit_stuff
        stuff_ratings = fit_stuff(pitches[pitches["game_type"].astype(str).eq("R")]).ratings
    F.stuff = stuff_ratings
    if stuff_ratings is not None and stuff_gamma:
        rates.stuff = stuff_ratings["rv100"].to_dict()
        rates.gamma = stuff_gamma
    return F


def rosters(cfg: Config, p: pd.DataFrame) -> dict:
    """Active hitters/pitchers per team and the simulated bullpens."""
    since = (p.loc[p["game_type"].astype(str).eq("R"), "game_date"].max() - pd.Timedelta(days=27)).strftime("%Y-%m-%d")
    out = {"since": since}
    for t in cfg.teams:
        pit, hit = active_players(p, t, since)
        rot = {int(g["starter"]) for g in cfg["rotation"][t]} | {int(g["bulk"]) for g in cfg["rotation"][t] if g.get("bulk")}
        bp = cfg["bullpen"][t]
        pen = [x for x in pit if x not in rot] if bp.get("mode", "auto") == "auto" else []
        pen = [x for x in pen if x not in set(bp.get("exclude", []))] + [int(x) for x in bp.get("include", [])]
        lineup_ids = {int(x) for v in cfg["lineups"][t].values() for x in v}
        late_pa = p[(p["game_date"] >= pd.Timestamp(since)) & p["bat_team"].eq(t) & p["pa_end"]].groupby("batter").size()
        hit = sorted(set(hit) | lineup_ids, key=lambda b: -late_pa.get(b, 0))
        out[t] = {"pitchers": sorted(set(pit) | rot, key=lambda x: -(p["pitcher"].eq(x) & p["fld_team"].eq(t)).sum()),
                  "hitters": hit, "bullpen": pen, "rotation": sorted(rot)}
    return out


def starter_candidates(cfg: Config, team: str) -> list[int]:
    seen = []
    for g in cfg["rotation"][team]:
        for k in ("starter", "bulk"):
            if g.get(k) and int(g[k]) not in seen:
                seen.append(int(g[k]))
    return seen


def bats_throws(p: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
    st = p.dropna(subset=["batter", "stand"]).groupby("batter")["stand"].agg(lambda s: "S" if s.nunique() > 1 else s.iloc[0])
    th = p.dropna(subset=["pitcher", "p_throws"]).groupby("pitcher")["p_throws"].first()
    return {int(k): str(v) for k, v in st.items()}, {int(k): str(v) for k, v in th.items()}


def matchup_matrix(cfg: Config, F: Fitted, ros: dict, bats: dict, throws: dict, lam: float, names: dict) -> dict:
    out = {}
    for bat_t in cfg.teams:
        fld_t = cfg.opponent(bat_t)
        hitters = list(dict.fromkeys(cfg["lineups"][bat_t]["vs_R"] + cfg["lineups"][bat_t]["vs_L"]))
        cands = starter_candidates(cfg, fld_t)
        rows = []
        for h in hitters:
            h = int(h)
            cells = []
            for pid in cands:
                hand = throws.get(pid, "R")
                side = batter_side(bats.get(h, "R"), hand)
                e = float(pair_edges(F.profiles, np.array([h]), np.array([pid]), np.array([side]))[0])
                probs = matchup_probs(F.rates, h, side, pid, hand, e, lam)
                base = matchup_probs(F.rates, h, side, pid, hand)
                cells.append({"p": pid, "xw": round(expected_woba(probs), 3), "xw_talent": round(expected_woba(base), 3),
                              "edge": round(e, 3), "probs": [round(float(x), 4) for x in probs],
                              "n_bat": round(F.rates.batter_n.get((h, hand), 0)), "n_pit": round(F.rates.pitcher_n.get((pid, side), 0)),
                              "detail": matchup_detail(F.profiles, h, pid, side), "side": side})
            rows.append({"h": h, "name": names.get(h, str(h)), "bats": bats.get(h, "R"), "cells": cells})
        out[bat_t] = {"pitchers": [{"id": pid, "name": names.get(pid, str(pid)), "th": throws.get(pid, "R")} for pid in cands],
                      "rows": rows}
    return out


def build_sim_context(cfg: Config, F: Fitted, ros: dict, bats: dict, throws: dict, lam: float, pf_mult: dict,
                      boost: float) -> tuple[SimContext, list[GamePlan]]:
    teams = {}
    p = F.pitches
    for t in cfg.teams:
        app = appearances(p, t, F.li)
        usage = UsageModel(app, ros[t]["bullpen"], boost=boost)
        starts = app[app["is_start"]]
        sbf = {int(pid): g["bf"].to_numpy() for pid, g in starts.groupby("pitcher")}
        # pitchers who rarely start (e.g. a bulk arm): use their longer outings
        for pid in starter_candidates(cfg, t):
            if pid not in sbf or len(sbf[pid]) < 3:
                long = app[app["pitcher"].eq(pid)]["bf"].to_numpy()
                sbf[pid] = long[long >= np.percentile(long, 50)] if len(long) else np.array([12])
        teams[t] = TeamSetup(t, {k: [int(x) for x in v] for k, v in cfg["lineups"][t].items()}, bats, throws,
                             ros[t]["bullpen"], usage, sbf, cfg.venue(t))
    # edges for every lineup batter vs every pitcher who might face him
    edges = {}
    for bat_t in cfg.teams:
        fld_t = cfg.opponent(bat_t)
        hitters = sorted({int(x) for v in cfg["lineups"][bat_t].values() for x in v})
        arms = sorted(set(ros[fld_t]["bullpen"]) | set(starter_candidates(cfg, fld_t)))
        B, P, S = [], [], []
        for h in hitters:
            for a in arms:
                B.append(h)
                P.append(a)
                S.append(batter_side(bats.get(h, "R"), throws.get(a, "R")))
        if B:
            e = pair_edges(F.profiles, np.array(B), np.array(P), np.array(S))
            edges.update({(b, a): float(x) for b, a, x in zip(B, P, e)})
    ctx = SimContext(F.rates, edges, lam, pf_mult, float(cfg.model.get("home_pa_tilt", 0.035)), teams)
    plans = []
    for i, g in enumerate(cfg["schedule"]):
        st = {t: cfg["rotation"][t][i] for t in cfg.teams}
        plans.append(GamePlan(g["home"], cfg.venue(g["home"]), st, date.fromisoformat(str(g["date"]))))
    return ctx, plans


def run(cfg: Config, n_series: int | None = None, n_boot: int | None = None, skip_backtest: bool = False,
        seed: int | None = None) -> dict:
    ensure_dirs()
    t0 = time.time()
    p, pa = load_prepared(cfg)
    _log(f"loaded {len(p):,} pitches, {len(pa):,} plate appearances", t0)
    m = cfg.model
    bt_path = OUTPUT / "backtest.json"
    bt = None
    if not skip_backtest:
        try:
            bt = backtest(p, pa, m)
            bt_path.write_text(json.dumps(bt, indent=2))
        except ValueError as e:
            print(f"backtest skipped: {e}")
    elif bt_path.exists():
        bt = json.loads(bt_path.read_text())
    tuned = bt.get("tuned") if bt else None
    stuff_gamma = float(bt["stuff"]["gamma"]) if bt and bt.get("stuff", {}).get("gain_pct", 0) > 0 else 0.0
    F = fit(cfg, p, pa, tuned, stuff_gamma)
    _log(f"stuff tilt gamma = {stuff_gamma}", t0)
    _log("fit rates and pitch profiles" + (f" (shrink x{tuned['shrink_mult']}, half-life {tuned['half_life']})" if tuned else ""), t0)
    lam = float(bt["lambda"]) if bt and bt.get("mix_gain_pct", 0) > 0 else 0.0
    _log(f"pitch-mix lambda = {lam}", t0)

    ros = rosters(cfg, p)
    bats, throws = bats_throws(p)
    ids = set()
    for t in cfg.teams:
        ids |= set(ros[t]["hitters"]) | set(ros[t]["pitchers"]) | set(ros[t]["bullpen"])
    names = player_names(p)
    names.update(lookup_names(ids))
    _log("rosters and names", t0)

    lb = {k: load_leaderboard(cfg, k) for k in ("oaa_team", "oaa_player", "sprint", "framing", "baserunning", "arm", "park_factors")}
    pf_mult = agg.park_multipliers(cfg, lb["park_factors"])
    boost = float(m.get("postseason_leverage_boost", 1.0))

    # ---- bullpen tendencies
    bull = {}
    for t in cfg.teams:
        app = appearances(p, t, F.li)
        reg = app[app["game_type"].astype(str).eq("R")] if "game_type" in app else app
        tend = reliever_tendencies(reg, names, ros[t]["bullpen"])
        sit = situational_splits(p, t, ros[t]["bullpen"], names)
        hooks = starter_hooks(reg)
        hooks = hooks[hooks.index.isin(starter_candidates(cfg, t))]
        u = UsageModel(reg, ros[t]["bullpen"], boost=1.0)
        usage_grid = {f"{k[0]}|{k[1]}": {str(pid): round(v, 3) for pid, v in d.items()} for k, d in u.table.items()}
        post = app[~app["game_type"].astype(str).eq("R")] if "game_type" in app else app.iloc[:0]
        bull[t] = {
            "tendencies": tend.reset_index().replace({np.nan: None}).round(3).to_dict(orient="records"),
            "situational": sit.replace({np.nan: None}).round(3).to_dict(orient="records"),
            "hooks": hooks.reset_index().round(2).to_dict(orient="records"),
            "usage_grid": usage_grid,
            "postseason_apps": post[["game_date", "pitcher", "entry_inning", "entry_margin", "bf", "pitches", "is_start"]]
            .assign(game_date=lambda x: x["game_date"].astype(str)).to_dict(orient="records"),
        }
    _log("bullpen tendencies", t0)

    # ---- matchups
    matrix = matchup_matrix(cfg, F, ros, bats, throws, lam, names)
    _log("hitter vs starter matrix", t0)

    # ---- simulation
    n_series = n_series or int(cfg.sim["n_series"])
    n_boot = int(cfg.sim.get("bootstrap", 0)) if n_boot is None else n_boot
    seed = int(cfg.sim.get("seed", 0)) if seed is None else seed
    ctx, plans = build_sim_context(cfg, F, ros, bats, throws, lam, pf_mult, boost)
    # run environment: league-average teams should score like the league did, adjusted for these parks
    reg_games = p[p["game_type"].astype(str).eq("R")].groupby("game_pk").agg(h=("post_home_score", "max"),
                                                                              a=("post_away_score", "max"))
    league_rpg = float((reg_games["h"].mean() + reg_games["a"].mean()) / 2)
    park_runs = agg.park_run_index(cfg, lb["park_factors"], [pl.venue for pl in plans])
    from .postseason_env import load_factor
    post_factor = load_factor(cfg)
    scoring = calibrate_scoring(ctx, plans, league_rpg * park_runs * post_factor, n_games=2000, seed=seed)
    scoring["post_factor"] = post_factor
    _log(f"scoring calibration: tilt {scoring['tilt']:+.3f}, league-average teams "
         f"{scoring['rpg_at_0']:.2f} -> {scoring['rpg_after']:.2f} R/G (target {scoring['target_rpg']:.2f})", t0)
    sim = sim_series(ctx, plans, n_series, seed)
    _log(f"simulated {n_series:,} series: " + ", ".join(f"{k} {v:.1%}" for k, v in sim["p_win"].items()), t0)
    boots = []
    if n_boot:
        games = p["game_pk"].unique()
        rng = np.random.default_rng(seed + 1)
        per = max(300, n_series // max(n_boot, 1))
        for b in range(n_boot):
            pick = rng.choice(games, size=len(games), replace=True)
            counts = pd.Series(pick).value_counts()
            # game-level bootstrap via weights: a game drawn twice counts twice in every rate
            pb = p[p["game_pk"].isin(counts.index)].assign(bw=lambda x: x["game_pk"].map(counts).astype(float))
            pab = pa[pa["game_pk"].isin(counts.index)].assign(bw=lambda x: x["game_pk"].map(counts).astype(float))
            Fb = fit(cfg, pb, pab, tuned, stuff_gamma, F.stuff)
            Fb.li = F.li
            cb, plb = build_sim_context(cfg, Fb, ros, bats, throws, lam, pf_mult, boost)
            cb.scoring_tilt = ctx.scoring_tilt
            r = sim_series(cb, plb, per, seed + 100 + b)
            boots.append(r["p_win"])
            _log(f"bootstrap {b + 1}/{n_boot}: " + ", ".join(f"{k} {v:.1%}" for k, v in r["p_win"].items()), t0)
    # ---- calibration: does the simulator score at the right level?
    cal = calibrate(ctx, plans, n_games=int(cfg.sim.get("calibration_games", 1500)), seed=seed)
    actual = {"league_rpg": league_rpg, "park_run_index": park_runs}
    for t in cfg.teams:
        G = agg.game_results(p[p["game_type"].astype(str).eq("R")], t)
        actual[t] = {"rpg": float(G["rf"].mean()), "rapg": float(G["ra"].mean())}
    calib = {"sim": cal, "actual": actual, "scoring": scoring}
    _log("calibration: sim league R/G " + ", ".join(f"{k} {v:.2f}" for k, v in cal["league"].items())
         + f" vs actual {actual['league_rpg']:.2f}; offense " + ", ".join(f"{k} {v:.2f}" for k, v in cal["offense"].items())
         + " vs actual " + ", ".join(f"{t} {actual[t]['rpg']:.2f}" for t in cfg.teams)
         + "; runs allowed " + ", ".join(f"{cfg.opponent(k)} {v:.2f}" for k, v in cal["defense"].items())
         + " vs actual " + ", ".join(f"{t} {actual[t]['rapg']:.2f}" for t in cfg.teams), t0)
    hs = cfg["higher_seed"]
    band = None
    if boots:
        v = np.array([x[hs] for x in boots])
        band = {"team": hs, "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
                "p90": float(np.percentile(v, 90)), "n": len(v)}
    sim["bootstrap"] = band
    sim["plans"] = [{"game": i + 1, "home": pl.home, "venue": pl.venue, "date": str(pl.day),
                     "starters": {t: {k: v for k, v in pl.starters[t].items()} for t in cfg.teams}} for i, pl in enumerate(plans)]

    # ---- descriptive aggregates (same structures as the first dashboard)
    tp = p[p["bat_team"].isin(cfg.teams) | p["fld_team"].isin(cfg.teams)]
    sched = load_schedule(cfg)
    data = {
        "meta": {"season": cfg.season, "as_of": str(F.as_of.date()), "generated": time.strftime("%Y-%m-%d %H:%M"),
                 "teams": {t: cfg.team_name(t) for t in cfg.teams}, "higher_seed": hs, "outcomes": OUTCOMES,
                 "lambda": lam, "n_pitches": int(len(p)), "scope": cfg.model.get("league_scope")},
        "team": {t: agg.team_tables(tp, t, ros[t]["pitchers"]) for t in cfg.teams},
        "hitters": {t: agg.hitter_cards(tp, t, ros[t]["hitters"], names, ros["since"]) for t in cfg.teams},
        "pitchers": {t: agg.pitcher_cards(tp, t, ros[t]["pitchers"], names, ros["since"]) for t in cfg.teams},
        "records": agg.records(tp, cfg),
        "conditions": agg.conditions(tp, sched, cfg),
        "defense": agg.defense(cfg, lb),
        "park": agg.park(cfg, lb["park_factors"]),
        "matrix": matrix, "bullpen": bull, "sim": sim, "backtest": bt, "calibration": calib,
        "rosters": {t: {k: v for k, v in ros[t].items()} for t in cfg.teams},
        "config": {"schedule": cfg["schedule"], "rotation": cfg["rotation"], "lineups": cfg["lineups"]},
    }
    if F.stuff is not None:
        sp = F.stuff["stuff_plus"].round(0).to_dict()
        for t in cfg.teams:
            for c in data["pitchers"][t]:
                c["stuff_plus"] = sp.get(int(c["id"]))
            for r in data["bullpen"][t]["tendencies"]:
                r["stuff_plus"] = sp.get(int(r["pitcher"]))
        data["meta"]["stuff_gamma"] = stuff_gamma
    _log("aggregates", t0)
    (OUTPUT / "dashboard_data.json").write_text(json.dumps(data, default=_json_default))
    render(data)
    _log(f"wrote {OUTPUT / 'dashboard_data.json'} and {DOCS / 'index.html'}", t0)
    return data


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp, date)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if o is pd.NA:
        return None
    return str(o)


def render(data: dict | None = None) -> None:
    if data is None:
        data = json.loads((OUTPUT / "dashboard_data.json").read_text())
    gb = OUTPUT / "game_backtest.json"
    if gb.exists():
        data["game_backtest"] = json.loads(gb.read_text())
    pe = OUTPUT / "postseason_env.json"
    if pe.exists():
        data["postseason_env"] = json.loads(pe.read_text())
    tmpl = DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    blob = json.dumps(data, default=_json_default, ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")
    DOCS.mkdir(parents=True, exist_ok=True)
    html = tmpl.replace("__DATA__", blob)
    if not html.lstrip().lower().startswith("<!doctype"):
        html = "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n" + html + "\n</html>"
    (DOCS / "index.html").write_text(html, encoding="utf-8")
