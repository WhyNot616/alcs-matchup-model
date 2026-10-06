"""Whole-postseason mode: simulate every remaining series, round by round, from the live bracket.

Inputs (all pulled automatically, all pregame information):
  - seeds from the regular-season standings, series state and schedule (including the placeholder
    LCS and World Series games) from the MLB Stats API
  - announced probable starters; after those run out, each team's rotation from its recent starts,
    choosing the most-rested starter with at least four days of rest
  - each team's most recent batting orders against right- and left-handed starters
  - each bullpen as it stood over the last 30 days, each manager's relief usage pattern, and real
    reliever fatigue from the last two days (Statcast plus boxscore pitch counts)
  - the outcome model, stuff ratings and October run environment exactly as in the backtests

Outputs: series odds, round-by-round odds for every team, the most likely LCS and World Series
matchups, and the next game in every series with the model's line next to the betting market's.
Optional per-team overrides live in config/playoffs.yaml.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .backtest_games import GameSpec, World, specs_from_statcast
from .config import DOCS, OUTPUT, ROOT, Config, ensure_dirs
from .pa_model import stuff_setting
from .simulate import Fatigue, GamePlan, SimContext, TeamSetup, sim_game

ROUND_OF = {"F": "WC", "D": "DS", "L": "LCS", "W": "WS"}
ROUND_ORDER = ["WC", "DS", "LCS", "WS"]
BEST_OF = {"WC": 3, "DS": 5, "LCS": 7, "WS": 7}
PLAYOFF_CONFIG = ROOT / "config" / "playoffs.yaml"
PLAYOFF_TEMPLATE = ROOT / "dashboard" / "playoffs.html"


def load_overrides() -> dict:
    if PLAYOFF_CONFIG.exists():
        return yaml.safe_load(PLAYOFF_CONFIG.read_text(encoding="utf-8")) or {}
    return {}


# ---------------------------------------------------------------- bracket state
ROUND_NAME = {"WC": "Wild Card Series", "DS": "Division Series", "LCS": "Championship Series", "WS": "World Series"}


def _league_hint(g: dict) -> str | None:
    for txt in (g.get("description") or "", g["home"].get("name") or "", g["away"].get("name") or ""):
        t = txt.strip().lower()
        if t.startswith(("al", "american")):
            return "AL"
        if t.startswith(("nl", "national")):
            return "NL"
    return None


def series_key(g: dict, teams: dict | None = None) -> str:
    """One series per league in the LCS and one World Series. Earlier rounds have several series per
    league, so their games are grouped by the (unordered) pair of teams or placeholder slots; the MLB
    description alone is not unique (both NL Wild Card series can carry the same one)."""
    rnd = ROUND_OF.get(g.get("game_type"), g.get("game_type"))
    if rnd == "WS":
        return "WS"
    if rnd == "LCS":
        lg = next((teams[g[s]["team"]]["league"] for s in ("home", "away") if teams and g[s]["team"] in teams), None)
        return f"LCS|{lg or _league_hint(g) or '?'}"
    ids = sorted((g[s]["team"] or (g[s].get("name") or "TBD")) for s in ("home", "away"))
    return f"{rnd}|{'-'.join(ids)}"


def build_state(br: dict) -> dict:
    teams = {t["team"]: t for t in br["teams"] if t.get("team")}
    seeds = {}
    for lg in ("AL", "NL"):
        L = [t for t in teams.values() if t["league"] == lg]
        div = sorted([t for t in L if str(t.get("division_rank")) == "1"], key=lambda t: (-t["pct"], -t["w"]))[:3]
        wc = sorted([t for t in L if str(t.get("wildcard_rank") or "").isdigit() and int(t["wildcard_rank"]) <= 3],
                    key=lambda t: int(t["wildcard_rank"]))
        for i, t in enumerate(div + wc):
            seeds[t["team"]] = i + 1
    groups = defaultdict(list)
    for g in br["games"]:
        if g.get("game_type") in ROUND_OF:
            groups[series_key(g, teams)].append(g)
    series = []
    for k, gs in groups.items():
        gs.sort(key=lambda g: (g.get("game_number") or 0, g["date"]))
        rnd = ROUND_OF[gs[0]["game_type"]]
        best_of = int(gs[0].get("games_in_series") or BEST_OF[rnd])
        real = sorted({g[s]["team"] for g in gs for s in ("home", "away") if g[s]["team"] in teams},
                      key=lambda t: (seeds.get(t, 99), -teams.get(t, {}).get("pct", 0)))
        league = None if rnd == "WS" else (teams[real[0]]["league"] if real else _league_hint(gs[0]))
        wins = Counter()
        for g in gs:
            if g["abstract"] == "Final" and g["home"]["score"] is not None and g["away"]["team"]:
                w = g["home"]["team"] if g["home"]["score"] > g["away"]["score"] else g["away"]["team"]
                wins[w] += 1
        need = best_of // 2 + 1
        winner = next((t for t, n in wins.items() if n >= need), None)
        label = ROUND_NAME[rnd] if rnd == "WS" else f"{league or ''} {ROUND_NAME[rnd]}".strip()
        series.append({"key": k, "round": rnd, "league": league, "label": label, "best_of": best_of, "need": need,
                       "teams": real if len(real) == 2 else None, "known": real, "wins": dict(wins),
                       "winner": winner, "games": gs})
    series.sort(key=lambda s: (ROUND_ORDER.index(s["round"]), s["league"] or "", min(
        [seeds.get(t, 99) for t in s["known"]] or [99])))
    return {"teams": teams, "seeds": seeds, "series": series}


def eliminated(state: dict) -> set[str]:
    return {t for s in state["series"] if s["winner"] for t in s["known"] if t != s["winner"]}


def alive_teams(state: dict) -> list[str]:
    """Every postseason team that has not lost a series (a World Series winner counts until the end)."""
    out = {t for s in state["series"] for t in s["known"]} - eliminated(state)
    return sorted(out)


def feeder(s: dict, state: dict) -> str | None:
    """For a Division Series with one known team (a bye seed), the Wild Card series that sends its
    opponent: seed 1 meets the 4 vs 5 winner, seed 2 the 3 vs 6 winner."""
    if s["round"] != "DS" or len(s["known"]) != 1:
        return None
    seed = state["seeds"].get(s["known"][0])
    want = {1: 4, 2: 3}.get(seed)
    for w in state["series"]:
        if w["round"] == "WC" and w["league"] == s["league"] and any(state["seeds"].get(t) == want for t in w["known"]):
            return w["key"]
    return None


def _higher(a: str, b: str, state: dict, rnd: str) -> tuple[str, str]:
    """Home-field order: seed within a league; better regular-season record in the World Series."""
    T, S = state["teams"], state["seeds"]
    if rnd == "WS":
        key = lambda t: (-T[t]["pct"], -T[t]["w"])  # noqa: E731
    else:
        key = lambda t: (S.get(t, 99), -T[t]["pct"])  # noqa: E731
    return tuple(sorted([a, b], key=key))  # type: ignore[return-value]


def _home_of(g: dict, hi: str, lo: str, state: dict) -> str:
    h = g["home"]
    if h["team"] in (hi, lo):
        return h["team"]
    name = (h.get("name") or "").lower()
    if "higher" in name:
        return hi
    if "lower" in name:
        return lo
    T = state["teams"]
    if name.startswith(("al ", "american")):
        return hi if T[hi]["league"] == "AL" else lo
    if name.startswith(("nl ", "national")):
        return hi if T[hi]["league"] == "NL" else lo
    n = g.get("game_number") or 1
    return hi if n in (1, 2, 6, 7) else lo


# ---------------------------------------------------------------- team inputs
def recent_games(p: pd.DataFrame, post_rows: list[dict], team: str, n: int = 12) -> list[dict]:
    """Most recent games for a team: lineup, starter, opposing starter hand. Boxscores first."""
    out = []
    for r in post_rows:
        for side, opp in (("home", "away"), ("away", "home")):
            if r[side]["team"] == team and len(r[side]["lineup"]) >= 9 and r[opp]["pitchers"]:
                out.append({"date": r["date"], "game_pk": int(r["game_pk"]), "lineup": r[side]["lineup"],
                            "starter": r[side]["pitchers"][0] if r[side]["pitchers"] else None,
                            "opp_starter": r[opp]["pitchers"][0]})
    seen = {g["game_pk"] for g in out}
    d = p[(p["home_team"].eq(team) | p["away_team"].eq(team))]
    pks = d.drop_duplicates("game_pk").sort_values("game_date")["game_pk"].tolist()[-n:]
    for s in specs_from_statcast(d, [x for x in pks if x not in seen]):
        out.append({"date": str(s.day), "game_pk": s.game_pk, "lineup": s.lineup[team], "starter": s.starter[team],
                    "opp_starter": s.starter[s.away if team == s.home else s.home]})
    return sorted(out, key=lambda g: (g["date"], g["game_pk"]))[-n:]


def team_inputs(team: str, games: list[dict], throws: dict, overrides: dict) -> dict:
    lu = {"vs_R": None, "vs_L": None}
    for g in reversed(games):
        h = throws.get(int(g["opp_starter"]), "R") if g["opp_starter"] else "R"
        if lu["vs_" + h] is None:
            lu["vs_" + h] = [int(x) for x in g["lineup"][:9]]
    last = [int(x) for x in games[-1]["lineup"][:9]] if games else []
    lu = {k: (v or last) for k, v in lu.items()}
    ov = (overrides.get("lineups") or {}).get(team) or {}
    lu.update({k: [int(x) for x in v] for k, v in ov.items() if v})
    starts = [(g["date"], int(g["starter"])) for g in games if g.get("starter")]
    rot = []
    for _, pid in reversed(starts):
        if pid not in rot:
            rot.append(pid)
        if len(rot) == 4:
            break
    rot = [int(x) for x in (overrides.get("rotation") or {}).get(team) or rot]
    last_start = {}
    for d, pid in starts:
        last_start[pid] = date.fromisoformat(d)
    return {"lineups": lu, "rotation": rot, "last_start": last_start}


def fatigue_seed(world: World, post_rows: list[dict], teams: list[str], day: date) -> list[tuple[int, date, int]]:
    seeds = world.fatigue(teams, day)
    have = {(pid, d) for pid, d, _ in seeds}
    statcast_pks = set(world.app_light["game_pk"].astype(int))
    for r in post_rows:
        d = date.fromisoformat(r["date"])
        if int(r["game_pk"]) in statcast_pks or not (day - timedelta(days=2) <= d < day):
            continue
        for side in ("home", "away"):
            if r[side]["team"] not in teams:
                continue
            for pid in r[side]["pitchers"][1:]:  # relievers only
                n = (r[side].get("pitch_counts") or {}).get(str(pid)) or 20
                if (pid, d) not in have:
                    seeds.append((int(pid), d, int(n)))
    return seeds


# ---------------------------------------------------------------- simulation
def simulate_bracket(cfg: Config, world: World, state: dict, inputs: dict, n: int, seed: int, boost: float,
                     post_rows: list[dict], excluded: set[int], verbose: bool = True) -> dict:
    teams_alive = alive_teams(state)
    setups = {}
    for t in teams_alive:
        if t not in inputs:
            continue
        rot = inputs[t]["rotation"]
        as_of = min((date.fromisoformat(g["date"]) for s in state["series"] for g in s["games"]
                     if g["abstract"] != "Final"), default=date.today())
        pen = [x for x in world.bullpen(t, as_of + timedelta(days=1), -1) if x not in rot and x not in excluded]
        usage = world.usage.get(t) or next(iter(world.usage.values()))
        usage.boost = boost
        setups[t] = TeamSetup(t, inputs[t]["lineups"], world.bats, world.throws, pen, usage, world.starter_bf,
                              world.venue_of.get(t, ""))
    ctx = SimContext(world.rates, {}, 0.0, world.park, world.home_tilt, setups, cache={},
                     scoring_tilt=world.scoring_tilt)
    first_open = min((date.fromisoformat(g["date"]) for s in state["series"] for g in s["games"]
                      if g["abstract"] != "Final"), default=date.today())
    fseed = fatigue_seed(world, post_rows, list(setups), first_open)
    rng = np.random.default_rng(seed)

    by_round = defaultdict(list)
    for s in state["series"]:
        by_round[s["round"]].append(s)
    tallies = {"series_win": defaultdict(Counter), "length": defaultdict(Counter), "round_win": defaultdict(Counter),
               "pairing": defaultdict(Counter), "game": defaultdict(lambda: {"played": 0, "home_win": 0, "runs_h": 0,
                                                                             "runs_a": 0, "starters": defaultdict(Counter)})}
    for _ in range(n):
        fat = Fatigue()
        for pid, d_, k in fseed:
            fat.add(pid, d_, k)
        last = {t: dict(inputs[t]["last_start"]) for t in setups}
        winners = {}
        for rnd in ROUND_ORDER:
            for s in by_round.get(rnd, []):
                if s["winner"]:
                    winners[s["key"]] = s["winner"]
                    continue
                if s["teams"]:
                    a, b = s["teams"]
                elif rnd == "DS" and feeder(s, state):
                    fk = feeder(s, state)
                    if fk not in winners:
                        continue
                    a, b = s["known"][0], winners[fk]
                elif rnd == "LCS":
                    w = [winners[x["key"]] for x in by_round["DS"] if x["league"] == s["league"] and x["key"] in winners]
                    if len(w) != 2:
                        continue
                    a, b = w
                elif rnd == "WS":
                    w = [winners[x["key"]] for x in by_round["LCS"] if x["key"] in winners]
                    if len(w) != 2:
                        continue
                    a, b = w
                else:
                    continue
                if a not in setups or b not in setups:
                    continue
                hi, lo = _higher(a, b, state, rnd)
                tallies["pairing"][s["key"]][f"{hi}-{lo}"] += 1
                wins = Counter(s["wins"])
                played_last = None
                for g in s["games"]:
                    if g["abstract"] == "Final":
                        continue
                    if max(wins[hi], wins[lo]) >= s["need"]:
                        break
                    home = _home_of(g, hi, lo, state)
                    away = lo if home == hi else hi
                    day = date.fromisoformat(g["date"])
                    st = {}
                    for t in (home, away):
                        pid = None
                        side = "home" if t == home else "away"
                        if g[side]["team"] == t and g[side].get("probable"):
                            pid = int(g[side]["probable"])
                        if pid is None:
                            pid = _pick_starter(inputs[t]["rotation"], last[t], day)
                        last[t][pid] = day
                        st[t] = {"starter": pid}
                    venue = g.get("venue") if g["home"]["team"] == home and any(k[0] == g.get("venue") for k in world.park) \
                        else world.venue_of.get(home, "")
                    ar, hr = sim_game(ctx, GamePlan(home, venue, st, day, away), fat, rng)
                    wins[home if hr > ar else away] += 1
                    gk = f"{s['key']}|{g.get('game_number')}"
                    gt = tallies["game"][gk]
                    gt["played"] += 1
                    gt["home_win"] += hr > ar
                    gt["runs_h"] += hr
                    gt["runs_a"] += ar
                    gt["starters"][home][st[home]["starter"]] += 1
                    gt["starters"][away][st[away]["starter"]] += 1
                    gt.setdefault("home_team", Counter())[home] += 1
                    played_last = g.get("game_number")
                w = hi if wins[hi] >= s["need"] else lo if wins[lo] >= s["need"] else None
                if w is None:
                    continue
                winners[s["key"]] = w
                tallies["series_win"][s["key"]][w] += 1
                tallies["length"][s["key"]][f"{w} in {played_last}"] += 1
                tallies["round_win"][rnd][w] += 1
    return tallies


def _pick_starter(rot: list[int], last: dict[int, date], day: date) -> int:
    if not rot:
        return -1
    def rest(p):
        return (day - last[p]).days if p in last else 99
    ready = [p for p in rot if rest(p) >= 5]
    pool = ready or rot
    return max(pool, key=lambda p: (rest(p), -rot.index(p)))


# ---------------------------------------------------------------- driver
def run(cfg: Config, n: int = 5000, seed: int = 11, verbose: bool = True) -> dict:
    from .data import (load_bracket, load_leaderboard, load_postseason_games, lookup_names, pull_bracket,
                       pull_espn_upcoming, pull_postseason_games)
    from .pipeline import load_prepared
    from .postseason_env import load_factor

    ensure_dirs()
    t0 = time.time()
    for f in (pull_bracket, pull_postseason_games):
        try:
            f(cfg)
        except Exception as e:
            print(f"  warning: {f.__name__} failed ({e})")
    br = load_bracket(cfg)
    if not br:
        raise RuntimeError("No bracket data. Run with network access first.")
    p, pa = load_prepared(cfg)
    post_rows = load_postseason_games(cfg)
    res = bracket(cfg, p, pa, br, post_rows, load_leaderboard(cfg, "park_factors"), n=n, seed=seed,
                  post_factor=load_factor(cfg), verbose=verbose)
    # market lines for the next game in each series
    nxt = [g for g in res["next_games"]]
    try:
        up = pull_espn_upcoming([g["date"] for g in nxt])
    except Exception as e:
        print(f"  warning: upcoming lines failed ({e})")
        up = {}
    for g in nxt:
        for r in up.values():
            if r["date"] == g["date"] and r["home"] == g["home"] and r["away"] == g["away"]:
                g["market"] = r["lines"].get("p_home_close")
                g["market_total"] = r["lines"].get("total_close")
                g["market_ml"] = r["lines"].get("ml_close")
    ids = {pid for g in nxt for pid in g["starters"].values() if pid and pid > 0}
    ids |= {pid for t in res["teams"].values() for pid in t.get("rotation", [])}
    ids |= {int(x[0]) for s in res["series"] for g in s["games"] for v in (g.get("starters") or {}).values() for x in v}
    ids |= {int(v) for s in res["series"] for g in s["games"] for v in (g.get("probables") or {}).values() if v}
    ids.discard(-1)
    names = lookup_names(ids)
    res["names"] = {str(k): v for k, v in names.items() if k in ids}
    (OUTPUT / "bracket.json").write_text(json.dumps(res, indent=1, default=str))
    render(res)
    print(f"[{time.time() - t0:6.1f}s] wrote {OUTPUT / 'bracket.json'} and {DOCS / 'index.html'}", flush=True)
    _report(res)
    return res


def _report(res: dict) -> None:
    """One-line summaries; on GitHub Actions they become run annotations."""
    gh = bool(__import__("os").environ.get("GITHUB_ACTIONS"))
    pre = "::notice title={}::" if gh else "{}: "
    T = res["teams"]
    top = sorted(T, key=lambda t: -T[t]["p_title"])
    print(pre.format("title odds") + ", ".join(f"{t} {T[t]['p_title']:.1%}" for t in top))
    print(pre.format("pennant odds") + ", ".join(f"{t} {T[t]['p_pennant']:.1%}" for t in top))
    for g in res["next_games"]:
        nm = res.get("names", {})
        st = " vs ".join(f"{nm.get(str(g['starters'][t]), g['starters'][t])}{'' if g['announced'].get(t) else ' (proj)'}"
                         for t in (g["away"], g["home"]))
        mk = f"; market {g['market']:.1%}, total {g.get('market_total')}" if g.get("market") is not None else "; no line"
        print(pre.format("next game") + f"{g['date']} {g['away']} @ {g['home']} G{g['game_number']}: {st}; model home "
              f"{g['p_home']:.1%}, runs {sum(g['exp_runs']):.1f}{mk}")
    s = res["settings"]
    print(pre.format("settings") + f"stuff {s['stuff_mode']} (gamma {s['stuff_gamma']}), post factor "
          f"{s['post_factor']:.3f}, scoring tilt {s['scoring_tilt']:.3f}, data through {res['as_of']}")


def bracket(cfg: Config, p: pd.DataFrame, pa: pd.DataFrame, br: dict, post_rows: list[dict],
            pf: pd.DataFrame | None, n: int = 5000, seed: int = 11, post_factor: float = 1.0,
            verbose: bool = True) -> dict:
    t0 = time.time()
    bt_path = OUTPUT / "backtest.json"
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else None
    tuned = bt.get("tuned") if bt else None
    sg = stuff_setting(bt)
    boost = float(cfg.model.get("postseason_leverage_boost", 1.0))
    ov = load_overrides()
    excluded = {int(x) for x in ov.get("exclude_pitchers", [])}

    state = build_state(br)
    world = World(cfg, p, p, pa, tuned, 0.0, pf, verbose, sg)
    alive = alive_teams(state)
    inputs = {}
    for t in alive:
        games = recent_games(p, post_rows, t)
        inputs[t] = team_inputs(t, games, world.throws, ov)
    # next game in each unfinished series with known teams (for calibration and the market check)
    nxt_specs = []
    for s in state["series"]:
        if s["winner"] or not s["teams"]:
            continue
        g = next((g for g in s["games"] if g["abstract"] != "Final"), None)
        if not g:
            continue
        home, away = g["home"]["team"], g["away"]["team"]
        if home not in inputs or away not in inputs:
            continue
        st = {}
        for t, side in ((home, "home"), (away, "away")):
            st[t] = int(g[side]["probable"]) if g[side].get("probable") else _pick_starter(
                inputs[t]["rotation"], inputs[t]["last_start"], date.fromisoformat(g["date"]))
        lu = {t: inputs[t]["lineups"]["vs_" + world.throws.get(st[o], "R")] for t, o in ((home, away), (away, home))}
        nxt_specs.append((s, g, GameSpec(int(g["game_pk"]), date.fromisoformat(g["date"]), g["game_type"], s["key"],
                                         home, away, lu, st, 0, 0, g.get("venue"))))
    if nxt_specs:
        world.calibrate_scoring([x[2] for x in nxt_specs], seed=seed, verbose=verbose, factor=post_factor)
    tallies = simulate_bracket(cfg, world, state, inputs, n, seed, boost, post_rows, excluded, verbose)
    if verbose:
        print(f"[{time.time() - t0:6.1f}s] simulated {n:,} brackets", flush=True)

    # ---- summarize
    T, S = state["teams"], state["seeds"]
    teams_out = {}
    for t in alive:
        rnd_p = {r: tallies["round_win"][r][t] / n for r in ROUND_ORDER}
        for s in state["series"]:  # rounds already won count as certain
            if s["winner"] == t:
                rnd_p[s["round"]] = 1.0
        in_wc = any(t in s["known"] for s in state["series"] if s["round"] == "WC")
        teams_out[t] = {"name": T[t]["name"], "league": T[t]["league"], "seed": S.get(t), "w": T[t]["w"], "l": T[t]["l"],
                        "p_wc": rnd_p["WC"] if in_wc else None, "p_ds": rnd_p["DS"], "p_pennant": rnd_p["LCS"], "p_title": rnd_p["WS"],
                        "rotation": inputs[t]["rotation"], "lineups": inputs[t]["lineups"]}
    series_out = []
    for s in state["series"]:
        o = {"key": s["key"], "round": s["round"], "league": s["league"], "label": s["label"], "best_of": s["best_of"],
             "teams": s["teams"], "known": s["known"], "feeder": feeder(s, state),
             "wins": s["wins"], "winner": s["winner"],
             "p_win": {k: v / n for k, v in tallies["series_win"][s["key"]].items()},
             "lengths": {k: v / n for k, v in tallies["length"][s["key"]].items()},
             "pairings": {k: v / n for k, v in tallies["pairing"][s["key"]].items()},
             "games": []}
        for g in s["games"]:
            row = {"game_number": g.get("game_number"), "date": g["date"], "status": g["status"],
                   "home": g["home"]["team"] or g["home"]["name"], "away": g["away"]["team"] or g["away"]["name"],
                   "score": [g["away"]["score"], g["home"]["score"]] if g["abstract"] == "Final" else None,
                   "probables": {"home": g["home"].get("probable"), "away": g["away"].get("probable")},
                   "if_necessary": g.get("if_necessary")}
            gt = tallies["game"].get(f"{s['key']}|{g.get('game_number')}")
            if gt and gt["played"]:
                row["p_played"] = gt["played"] / n
                row["p_home_win"] = gt["home_win"] / gt["played"]
                row["exp_runs"] = [gt["runs_a"] / gt["played"], gt["runs_h"] / gt["played"]]
                row["starters"] = {t: [[int(pid), c / gt["played"]] for pid, c in cnt.most_common(2)]
                                   for t, cnt in gt["starters"].items()}
                if gt.get("home_team"):
                    row["home_mix"] = {t: c / gt["played"] for t, c in gt["home_team"].items()}
            o["games"].append(row)
        series_out.append(o)
    next_games = []
    for s, g, spec in nxt_specs:
        gt = tallies["game"].get(f"{s['key']}|{g.get('game_number')}")
        if not gt or not gt["played"]:
            continue
        next_games.append({"series": s["key"], "game_number": g.get("game_number"), "date": g["date"],
                           "home": spec.home, "away": spec.away, "starters": spec.starter,
                           "announced": {spec.home: bool(g["home"].get("probable")), spec.away: bool(g["away"].get("probable"))},
                           "p_home": gt["home_win"] / gt["played"],
                           "exp_runs": [gt["runs_a"] / gt["played"], gt["runs_h"] / gt["played"]],
                           "status": g["status"]})
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "as_of": str(world.as_of.date()), "n": n,
           "settings": {"post_factor": post_factor, "stuff_mode": sg[0], "stuff_gamma": sg[1], "tuned": tuned, "boost": boost,
                        "scoring_tilt": world.scoring_tilt},
           "seeds": S, "teams": teams_out, "series": series_out, "next_games": next_games,
           "eliminated": sorted(eliminated(state)),
           "team_info": {t: {"name": T[t]["name"], "league": T[t]["league"], "seed": S.get(t), "w": T[t]["w"], "l": T[t]["l"]}
                         for t in sorted({x for s in state["series"] for x in s["known"]})}}
    return out


def render(res: dict | None = None) -> None:
    if res is None:
        res = json.loads((OUTPUT / "bracket.json").read_text())
    data = {"bracket": res}
    for f in ("game_backtest.json", "postseason_env.json"):
        path = OUTPUT / f
        if path.exists():
            data[f.split(".")[0]] = json.loads(path.read_text())
    if (OUTPUT / "dashboard_data.json").exists():
        dd = json.loads((OUTPUT / "dashboard_data.json").read_text())
        data["series_page"] = {"teams": dd["meta"]["teams"], "p_win": dd["sim"]["p_win"]}
    blob = json.dumps(data, default=str, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = PLAYOFF_TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", blob)
    if not html.lstrip().lower().startswith("<!doctype"):
        html = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
                "content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n" + html + "\n</html>")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    Path(DOCS / ".nojekyll").touch()
