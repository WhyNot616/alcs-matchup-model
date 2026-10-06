"""Data pulls from Baseball Savant and the MLB Stats API, cached as parquet/CSV.

Savant's Statcast search caps each CSV download at 25,000 rows, so pulls are split into
date windows and any window that hits the cap is split again.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_RAW, Config, ensure_dirs

SAVANT = "https://baseballsavant.mlb.com"
STATSAPI = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "alcs-matchup-model/0.1 (research project)"}
ROW_CAP = 25_000
TEAM_IDS = {"AZ": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112, "CWS": 145, "CIN": 113, "CLE": 114,
            "COL": 115, "DET": 116, "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158,
            "MIN": 142, "NYM": 121, "NYY": 147, "ATH": 133, "PHI": 143, "PIT": 134, "SD": 135, "SF": 137,
            "SEA": 136, "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120}
GAME_TYPES = {"regular": "R", "wildcard": "F", "division": "D", "league": "L", "world": "W"}


def _get(url: str, params: dict | None = None, retries: int = 4, timeout: int = 180) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {r.url}")
        except requests.RequestException as e:  # network hiccup, retry
            last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {last}")


def _statcast_window(season: int, start: date, end: date, game_types: str,
                     team: str | None, player_type: str) -> pd.DataFrame:
    params = {
        "all": "true", "hfGT": game_types, "hfSea": f"{season}|", "player_type": player_type,
        "game_date_gt": start.isoformat(), "game_date_lt": end.isoformat(),
        "min_pitches": 0, "min_results": 0, "group_by": "name", "sort_col": "pitches",
        "sort_order": "desc", "min_pas": 0, "type": "details",
    }
    if team:
        params["team"] = team
    text = _get(f"{SAVANT}/statcast_search/csv", params).text.lstrip("﻿")
    if not text.strip() or text.lstrip().startswith("<"):
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    if len(df) >= ROW_CAP - 5:
        if start >= end:
            raise RuntimeError(f"Single day {start} exceeds the Savant row cap; narrow the query.")
        mid = start + (end - start) // 2
        return pd.concat([
            _statcast_window(season, start, mid, game_types, team, player_type),
            _statcast_window(season, mid + timedelta(days=1), end, game_types, team, player_type),
        ], ignore_index=True)
    return df


def _date_windows(start: date, end: date, days: int):
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def statcast_path(cfg: Config, scope: str) -> Path:
    return DATA_RAW / f"statcast_{scope}_{cfg.season}.parquet"


def pull_statcast(cfg: Config, scope: str = "league", refresh: bool = False,
                  include_postseason: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Pull every pitch for the season.

    scope="league": all games (about 750k pitches, needed for league baselines and the backtest).
    scope="teams": only games involving the two configured teams (about 50k pitches, much faster).
    Incremental: if the cache exists and refresh=False, only dates from the last cached day onward
    are pulled and merged.
    """
    ensure_dirs()
    path = statcast_path(cfg, scope)
    season = cfg.season
    types = "R|" + ("F|D|L|W|" if include_postseason else "")
    start = date(season, 3, 1)
    end = min(date.today(), date(season, 11, 15))
    old = None
    if path.exists() and not refresh:
        old = pd.read_parquet(path)
        last = pd.to_datetime(old["game_date"]).max().date()
        start = last  # re-pull the last cached day in case it was partial
        old = old[pd.to_datetime(old["game_date"]).dt.date < last]
        if verbose:
            print(f"cache has {len(old):,} pitches through {last}; pulling {start} to {end}")
    frames = []
    if scope == "league":
        windows = list(_date_windows(start, end, 4))
        for i, (a, b) in enumerate(windows, 1):
            df = _statcast_window(season, a, b, types, None, "pitcher")
            if verbose:
                print(f"  [{i}/{len(windows)}] {a} to {b}: {len(df):,} pitches")
            frames.append(df)
    elif scope == "teams":
        for team in cfg.teams:
            for ptype in ("batter", "pitcher"):
                for a, b in _date_windows(start, end, 45):
                    df = _statcast_window(season, a, b, types, team, ptype)
                    if verbose:
                        print(f"  {team} {ptype} {a} to {b}: {len(df):,} pitches")
                    frames.append(df)
    else:
        raise ValueError("scope must be 'league' or 'teams'")
    new = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
    out = pd.concat([old, new], ignore_index=True) if old is not None else new
    if out.empty:
        raise RuntimeError("No Statcast rows returned. Check the season/date range or Savant availability.")
    out = out.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"], keep="last")
    out = out.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    # Parquet needs consistent dtypes; object columns with mixed types become strings.
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype("string")
    out.to_parquet(path, index=False)
    if verbose:
        print(f"saved {len(out):,} pitches to {path}")
    return out


def load_statcast(cfg: Config, scope: str | None = None) -> pd.DataFrame:
    scope = scope or cfg.model.get("league_scope", "league")
    path = statcast_path(cfg, scope)
    if not path.exists():
        alt = statcast_path(cfg, "teams" if scope == "league" else "league")
        if alt.exists():
            print(f"note: {path.name} not found, using {alt.name}")
            path = alt
        else:
            raise FileNotFoundError(f"{path} not found. Run `alcs pull` first.")
    return pd.read_parquet(path)


# ---------------------------------------------------------------- leaderboards
LEADERBOARDS = {
    "oaa_team": "/leaderboard/outs_above_average?type=Fielding_Team&startYear={y}&endYear={y}&split=no&team=&range=year&min=q&pos=&roles=&viz=hide&csv=true",
    "oaa_player": "/leaderboard/outs_above_average?type=Fielder&startYear={y}&endYear={y}&split=no&team=&range=year&min=1&pos=&roles=&viz=hide&csv=true",
    "sprint": "/leaderboard/sprint_speed?min_season={y}&max_season={y}&position=&team=&min=10&csv=true",
    "framing": "/leaderboard/catcher-framing?type=catcher&seasonStart={y}&seasonEnd={y}&team=&min=q&sortColumn=rv_tot&sortDirection=desc&csv=true",
    "baserunning": "/leaderboard/baserunning-run-value?game_type=Regular&season_start={y}&season_end={y}&sortColumn=runner_runs_tot&sortDirection=desc&split=no&n=0&team=&type=Team&csv=true",
    "arm": "/leaderboard/arm-strength?type=team&year={y}&minThrows=50&pos=&team=&csv=true",
}


def pull_leaderboards(cfg: Config, verbose: bool = True) -> dict[str, Path]:
    ensure_dirs()
    out = {}
    for key, tmpl in LEADERBOARDS.items():
        path = DATA_RAW / f"{key}_{cfg.season}.csv"
        try:
            text = _get(SAVANT + tmpl.format(y=cfg.season)).text.lstrip("﻿")
            if text.lstrip().startswith("<"):
                raise RuntimeError("got HTML instead of CSV")
            path.write_text(text, encoding="utf-8")
            out[key] = path
            if verbose:
                print(f"  {key}: {len(text.splitlines()) - 1} rows")
        except Exception as e:  # leaderboards are optional inputs
            print(f"  warning: {key} failed ({e})")
    # Park factors are embedded in the page as `var data = [...]`
    rows = []
    for years in (1, 3):
        for side in ("", "L", "R"):
            url = (f"{SAVANT}/leaderboard/statcast-park-factors?type=year&year={cfg.season}&batSide={side}"
                   f"&stat=index_wOBA&condition=All&rolling={years}&parks=mlb")
            try:
                html = _get(url).text
                m = re.search(r"var data = (\[.*?\]);", html, re.S)
                for r in json.loads(m.group(1)) if m else []:
                    r["rolling_years"] = years
                    r["bat_side_key"] = side or "All"
                    rows.append(r)
            except Exception as e:
                print(f"  warning: park factors {years}y {side or 'All'} failed ({e})")
    if rows:
        path = DATA_RAW / f"park_factors_{cfg.season}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        out["park_factors"] = path
        if verbose:
            print(f"  park_factors: {len(rows)} rows")
    return out


def load_leaderboard(cfg: Config, key: str) -> pd.DataFrame | None:
    path = DATA_RAW / f"{key}_{cfg.season}.csv"
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------- schedule
def pull_schedule(cfg: Config, verbose: bool = True) -> Path:
    """Regular-season and postseason games for both teams, with day/night, venue and weather."""
    ensure_dirs()
    rows = []
    for abbr in cfg.teams:
        tid = TEAM_IDS[abbr]
        url = f"{STATSAPI}/schedule"
        params = {"sportId": 1, "teamId": tid, "season": cfg.season, "gameType": "R,F,D,L,W",
                  "hydrate": "weather,probablePitcher,linescore"}
        js = _get(url, params).json()
        for d in js.get("dates", []):
            for g in d["games"]:
                w = g.get("weather") or {}
                rows.append({
                    "team": abbr, "game_pk": g["gamePk"], "game_date": d["date"], "game_type": g["gameType"],
                    "day_night": g.get("dayNight"), "venue": g["venue"]["name"],
                    "temp": pd.to_numeric(w.get("temp"), errors="coerce"), "condition": w.get("condition"),
                    "status": g["status"]["detailedState"],
                    "home": g["teams"]["home"]["team"]["name"], "away": g["teams"]["away"]["team"]["name"],
                    "home_score": g["teams"]["home"].get("score"), "away_score": g["teams"]["away"].get("score"),
                    "series": g.get("seriesDescription"),
                })
    path = DATA_RAW / f"schedule_{cfg.season}.csv"
    pd.DataFrame(rows).drop_duplicates(["team", "game_pk"]).to_csv(path, index=False)
    if verbose:
        print(f"  schedule: {len(rows)} rows")
    return path


def load_schedule(cfg: Config) -> pd.DataFrame | None:
    path = DATA_RAW / f"schedule_{cfg.season}.csv"
    return pd.read_csv(path) if path.exists() else None


def pull_all(cfg: Config, scope: str | None = None, refresh: bool = False) -> None:
    scope = scope or cfg.model.get("league_scope", "league")
    t0 = datetime.now()
    print(f"Pulling Statcast ({scope})...")
    pull_statcast(cfg, scope=scope, refresh=refresh)
    print("Pulling leaderboards...")
    pull_leaderboards(cfg)
    print("Pulling schedule...")
    pull_schedule(cfg)
    print(f"done in {(datetime.now() - t0).seconds}s")


# ---------------------------------------------------------------- player names
def lookup_names(ids, cache_path: Path | None = None) -> dict[int, str]:
    """MLBAM id -> 'Last, First' via the MLB Stats API, cached on disk.

    Set the environment variable ALCS_OFFLINE=1 to skip the network (ids are used as names).
    """
    cache_path = cache_path or (DATA_RAW / "people.json")
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    missing = [int(i) for i in set(int(x) for x in ids) if str(int(i)) not in cache]
    if os.environ.get("ALCS_OFFLINE"):
        missing = []
    for i in range(0, len(missing), 150):
        chunk = missing[i:i + 150]
        try:
            js = _get(f"{STATSAPI}/people", {"personIds": ",".join(map(str, chunk))}).json()
            for p in js.get("people", []):
                last, first = p.get("lastName", ""), p.get("useName") or p.get("firstName", "")
                cache[str(p["id"])] = f"{last}, {first}".strip(", ")
        except Exception as e:  # names are cosmetic; fall back to ids
            print(f"  warning: name lookup failed ({e})")
            break
    if missing:
        ensure_dirs()
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {int(k): v for k, v in cache.items()}
