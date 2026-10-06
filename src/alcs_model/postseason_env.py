"""How much less (or more) do teams score in the postseason than their regular-season numbers predict?

For each past season, every postseason game gets an expected score from the two teams' regular-season
runs scored and allowed per game (the same log5-style formula analysts use for run expectancy):

    expected runs for team A vs B = RS/G(A) * RA/G(B) / league R/G

The postseason run factor is total actual runs divided by total expected runs across all past
postseason games, with a bootstrap interval over games. Because playoff teams are better than
average on both sides, comparing raw postseason scoring with the regular season would mix team
quality with the October environment (aces pitching more, top relievers in more innings, cold
weather). This isolates the environment.

The factor is fit only on seasons before the current one, so applying it to this postseason is
leak-free.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import OUTPUT, Config, ensure_dirs


def expected_runs(scores: pd.DataFrame) -> pd.DataFrame:
    """Postseason games of one or more seasons with expected runs from regular-season team rates."""
    out = []
    for season, s in scores.groupby("season"):
        reg = s[s["game_type"].eq("R")]
        post = s[~s["game_type"].eq("R")].copy()
        if reg.empty or post.empty:
            continue
        rs = pd.concat([reg.groupby("home")["home_runs"].sum(), reg.groupby("away")["away_runs"].sum()]).groupby(level=0).sum()
        ra = pd.concat([reg.groupby("home")["away_runs"].sum(), reg.groupby("away")["home_runs"].sum()]).groupby(level=0).sum()
        g = pd.concat([reg["home"].value_counts(), reg["away"].value_counts()]).groupby(level=0).sum()
        rs_g, ra_g = rs / g, ra / g
        lg = (reg["home_runs"].sum() + reg["away_runs"].sum()) / (2 * len(reg))
        post["exp_home"] = post["home"].map(rs_g) * post["away"].map(ra_g) / lg
        post["exp_away"] = post["away"].map(rs_g) * post["home"].map(ra_g) / lg
        post["league_rpg"] = lg
        post["reg_rpg_teams"] = (post["home"].map(rs_g) + post["away"].map(rs_g)) / 2
        out.append(post.dropna(subset=["exp_home", "exp_away"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fit_factor(post: pd.DataFrame, n_boot: int = 4000, seed: int = 0) -> dict:
    act = (post["home_runs"] + post["away_runs"]).to_numpy(float)
    exp = (post["exp_home"] + post["exp_away"]).to_numpy(float)
    f = act.sum() / exp.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(act), size=(n_boot, len(act)))
    boots = act[idx].sum(1) / exp[idx].sum(1)
    by_season = []
    for season, s in post.groupby("season"):
        a = (s["home_runs"] + s["away_runs"]).sum()
        e = (s["exp_home"] + s["exp_away"]).sum()
        by_season.append({"season": int(season), "games": int(len(s)), "actual_rpg": float(a / (2 * len(s))),
                          "expected_rpg": float(e / (2 * len(s))), "league_rpg": float(s["league_rpg"].iloc[0]),
                          "factor": float(a / e)})
    by_round = []
    for gt, s in post.groupby("game_type"):
        a = (s["home_runs"] + s["away_runs"]).sum()
        e = (s["exp_home"] + s["exp_away"]).sum()
        by_round.append({"game_type": str(gt), "games": int(len(s)), "factor": float(a / e)})
    return {"factor": float(f), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "games": int(len(act)), "actual_rpg": float(act.sum() / (2 * len(act))),
            "expected_rpg": float(exp.sum() / (2 * len(act))), "by_season": by_season, "by_round": by_round}


def run(cfg: Config, seasons: list[int] | None = None, verbose: bool = True) -> dict:
    from .data import pull_season_scores

    ensure_dirs()
    seasons = seasons or list(cfg.model.get("postseason_env_seasons", [2021, 2022, 2023, 2024, 2025]))
    frames = [pull_season_scores(y, verbose=verbose) for y in seasons]
    post = expected_runs(pd.concat(frames, ignore_index=True))
    res = fit_factor(post)
    res["seasons"] = seasons
    # same check on the current postseason so far (not used for fitting)
    try:
        cur = expected_runs(pull_season_scores(cfg.season, refresh=True, verbose=verbose))
        if len(cur):
            c = fit_factor(cur, n_boot=2000)
            res["current"] = {k: c[k] for k in ("factor", "lo", "hi", "games", "actual_rpg", "expected_rpg")}
    except Exception as e:
        print(f"  warning: current-season check failed ({e})")
    (OUTPUT / "postseason_env.json").write_text(json.dumps(res, indent=1))
    if verbose:
        print(f"Postseason run factor {res['factor']:.3f} (95% {res['lo']:.3f} to {res['hi']:.3f}) over "
              f"{res['games']} games, {seasons[0]} to {seasons[-1]}")
        for s in res["by_season"]:
            print(f"  {s['season']}: {s['games']} games, actual {s['actual_rpg']:.2f} vs expected "
                  f"{s['expected_rpg']:.2f} R/G per team, factor {s['factor']:.3f}")
        if "current" in res:
            c = res["current"]
            print(f"  {cfg.season} so far: {c['games']} games, factor {c['factor']:.3f} ({c['lo']:.3f} to {c['hi']:.3f})")
    return res


def load_factor(cfg: Config) -> float:
    """The factor to use: config number, or the fitted value when the config says 'auto'."""
    v = cfg.model.get("postseason_run_factor", "auto")
    if isinstance(v, (int, float)):
        return float(v)
    path = OUTPUT / "postseason_env.json"
    if path.exists():
        return float(json.loads(path.read_text())["factor"])
    return 1.0
