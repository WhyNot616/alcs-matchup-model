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

from .config import DATA_RAW, ROOT, Config, ensure_dirs

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
                print(f"  [{i}/{len(windows)}] {a} to {b}: {len(df):,} pitches", flush=True)
            if os.environ.get("GITHUB_ACTIONS") and (i % 10 == 0 or i == len(windows)):
                print(f"::notice title=pull progress::window {i}/{len(windows)} through {b}", flush=True)
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
def search_people(query: str, limit: int = 10) -> list[tuple[int, str, str]]:
    """Find MLBAM ids by name: local caches first (people.json, the bracket), then the MLB Stats API.
    Returns (id, 'Last, First', source) rows."""
    q = query.lower().replace(",", " ").split()
    hits: dict[int, tuple[str, str]] = {}

    def check(pid, name, src):
        n = str(name).lower().replace(",", " ")
        if all(w in n for w in q):
            hits.setdefault(int(pid), (str(name), src))
    cache = DATA_RAW / "people.json"
    if cache.exists():
        for k, v in json.loads(cache.read_text(encoding="utf-8")).items():
            check(k, v, "cache")
    bj = ROOT / "output" / "bracket.json"
    if bj.exists():
        for k, v in (json.loads(bj.read_text(encoding="utf-8")).get("names") or {}).items():
            check(k, v, "bracket")
    if len(hits) < limit and not os.environ.get("ALCS_OFFLINE"):
        try:
            js = _get(f"{STATSAPI}/people/search", {"names": query, "sportIds": 1}).json()
            for p in js.get("people", []):
                last, first = p.get("lastName", ""), p.get("useName") or p.get("firstName", "")
                team = (p.get("currentTeam") or {}).get("id")
                hits.setdefault(int(p["id"]), (f"{last}, {first}", f"MLB API{f', team {ID2ABBR.get(team, team)}' if team else ''}"))
        except Exception as e:
            print(f"  warning: MLB people search failed ({e})")
    return [(k, v[0], v[1]) for k, v in list(hits.items())[:limit]]


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


# ---------------------------------------------------------------- postseason boxscores
ID2ABBR = {v: k for k, v in TEAM_IDS.items()}


def pull_postseason_games(cfg: Config, verbose: bool = True) -> Path | None:
    """Final postseason games with batting orders and starters from MLB boxscores.

    Used by the game backtest so playoff games can be scored even before Savant has their pitches.
    """
    ensure_dirs()
    if os.environ.get("ALCS_OFFLINE"):
        return None
    js = _get(f"{STATSAPI}/schedule", {"sportId": 1, "season": cfg.season, "gameType": "F,D,L,W"}).json()
    rows = []
    for d in js.get("dates", []):
        for g in d["games"]:
            st = g.get("status", {})
            if st.get("abstractGameState") != "Final" or st.get("detailedState") in ("Postponed", "Cancelled"):
                continue
            pk = g["gamePk"]
            try:
                box = _get(f"{STATSAPI}/game/{pk}/boxscore").json()
            except Exception as e:
                print(f"  warning: boxscore {pk} failed ({e})")
                continue

            def side(s):
                t = box["teams"][s]
                players = t.get("players", {})
                counts = {}
                for pid in t.get("pitchers", []):
                    st = ((players.get(f"ID{pid}") or {}).get("stats") or {}).get("pitching") or {}
                    counts[str(pid)] = st.get("numberOfPitches") or st.get("pitchesThrown")
                return {"team": ID2ABBR.get(int(t["team"]["id"])),
                        "lineup": [int(x) for x in t.get("battingOrder", [])][:9],
                        "pitchers": [int(x) for x in t.get("pitchers", [])], "pitch_counts": counts}
            rows.append({
                "game_pk": int(pk), "date": d["date"], "game_type": g.get("gameType"),
                "label": f"{g.get('seriesDescription', '')} G{g.get('seriesGameNumber', '')}".strip(),
                "venue": g["venue"]["name"], "home": side("home"), "away": side("away"),
                "home_runs": g["teams"]["home"].get("score"), "away_runs": g["teams"]["away"].get("score"),
            })
    path = DATA_RAW / f"postseason_games_{cfg.season}.json"
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    if verbose:
        print(f"  postseason games: {len(rows)} final")
    return path


def load_postseason_games(cfg: Config) -> list[dict]:
    path = DATA_RAW / f"postseason_games_{cfg.season}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


# ---------------------------------------------------------------- past seasons' scores
def pull_season_scores(season: int, refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Final scores of every regular-season and postseason game in a season (MLB Stats API)."""
    ensure_dirs()
    path = DATA_RAW / f"scores_{season}.csv"
    if path.exists() and not refresh and season < date.today().year:
        return pd.read_csv(path)
    rows = []
    for gt in ("R", "F,D,L,W"):
        js = _get(f"{STATSAPI}/schedule", {"sportId": 1, "season": season, "gameType": gt}).json()
        for d in js.get("dates", []):
            for g in d["games"]:
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                h, a = g["teams"]["home"], g["teams"]["away"]
                if h.get("score") is None or a.get("score") is None:
                    continue
                rows.append({"season": season, "game_pk": g["gamePk"], "date": d["date"], "game_type": g["gameType"],
                             "home": ID2ABBR.get(int(h["team"]["id"])), "away": ID2ABBR.get(int(a["team"]["id"])),
                             "home_runs": int(h["score"]), "away_runs": int(a["score"])})
    df = pd.DataFrame(rows).drop_duplicates("game_pk")
    df.to_csv(path, index=False)
    if verbose:
        n_post = int((df["game_type"] != "R").sum()) if len(df) else 0
        print(f"  {season}: {len(df) - n_post} regular-season and {n_post} postseason games")
    return df


# ---------------------------------------------------------------- betting lines (ESPN / DraftKings)
ESPN = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
ESPN_ABBR = {"ARI": "AZ", "CHW": "CWS", "WSN": "WSH", "OAK": "ATH", "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB"}


def _ml_prob(ml) -> float | None:
    try:
        v = float(str(ml).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if v == 0:
        return None
    return -v / (-v + 100) if v < 0 else 100 / (v + 100)


def _parse_pick(pick: list) -> dict:
    """Closing (else opening) moneyline and total from ESPN's pickcenter block."""
    if not pick:
        return {}
    p = pick[0]
    out = {"provider": (p.get("provider") or {}).get("name")}
    ml = p.get("moneyline") or {}
    for when in ("close", "open"):
        h = ((ml.get("home") or {}).get(when) or {}).get("odds")
        a = ((ml.get("away") or {}).get(when) or {}).get("odds")
        ph, pa_ = _ml_prob(h), _ml_prob(a)
        if ph and pa_:
            out[f"p_home_{when}"] = ph / (ph + pa_)        # proportional vig removal
            out[f"ml_{when}"] = [h, a]
    if "p_home_close" not in out:
        h = (p.get("homeTeamOdds") or {}).get("moneyLine")
        a = (p.get("awayTeamOdds") or {}).get("moneyLine")
        ph, pa_ = _ml_prob(h), _ml_prob(a)
        if ph and pa_:
            out["p_home_close"] = ph / (ph + pa_)
    tot = ((p.get("total") or {}).get("over") or {})
    for when in ("close", "open"):
        line = (tot.get(when) or {}).get("line")
        if line:
            try:
                out[f"total_{when}"] = float(str(line).lstrip("ou"))
            except ValueError:
                pass
    if "total_close" not in out and p.get("overUnder") is not None:
        out["total_close"] = float(p["overUnder"])
    return out


def pull_espn_odds(dates, season: int, verbose: bool = True) -> dict:
    """Pregame lines for every game on the given dates. Cached by ESPN event id in data/raw."""
    ensure_dirs()
    path = DATA_RAW / f"espn_odds_{season}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    if os.environ.get("ALCS_OFFLINE"):
        return cache
    new = 0
    for d in sorted({pd.Timestamp(x).strftime("%Y%m%d") for x in dates}):
        try:
            sb = _get(f"{ESPN}/scoreboard", {"dates": d, "limit": 50}).json()
        except Exception as e:
            print(f"  warning: ESPN scoreboard {d} failed ({e})")
            continue
        for ev in sb.get("events", []):
            eid = str(ev["id"])
            comp = ev["competitions"][0]
            if (comp.get("status", {}).get("type", {}).get("name") != "STATUS_FINAL") or (eid in cache and cache[eid].get("lines")):
                continue
            teams = {c["homeAway"]: c["team"]["abbreviation"] for c in comp["competitors"]}
            rec = {"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "start": ev.get("date"),
                   "home": ESPN_ABBR.get(teams.get("home"), teams.get("home")),
                   "away": ESPN_ABBR.get(teams.get("away"), teams.get("away")),
                   "scores": {c["homeAway"]: c.get("score") for c in comp["competitors"]}}
            try:
                s = _get(f"{ESPN}/summary", {"event": eid}).json()
                rec["lines"] = _parse_pick(s.get("pickcenter") or [])
            except Exception as e:
                rec["lines"] = {}
                print(f"  warning: ESPN summary {eid} failed ({e})")
            cache[eid] = rec
            new += 1
        path.write_text(json.dumps(cache))
    if verbose:
        have = sum(1 for v in cache.values() if v.get("lines", {}).get("p_home_close"))
        print(f"  ESPN odds: {new} new events, {have} of {len(cache)} cached events have a moneyline")
    return cache


# ---------------------------------------------------------------- live bracket
LEAGUE_IDS = {103: "AL", 104: "NL"}


def pull_bracket(cfg: Config, verbose: bool = True) -> Path | None:
    """Standings (for seeds) and every postseason game, played or scheduled, with probable pitchers."""
    ensure_dirs()
    if os.environ.get("ALCS_OFFLINE"):
        return None
    teams = []
    for lid, lg in LEAGUE_IDS.items():
        js = _get(f"{STATSAPI}/standings", {"leagueId": lid, "season": cfg.season,
                                             "standingsTypes": "regularSeason"}).json()
        for rec in js.get("records", []):
            for t in rec.get("teamRecords", []):
                teams.append({"team": ID2ABBR.get(int(t["team"]["id"])), "team_id": int(t["team"]["id"]),
                              "name": t["team"].get("name"), "league": lg, "w": int(t["wins"]), "l": int(t["losses"]),
                              "pct": float(t["winningPercentage"]), "division_rank": t.get("divisionRank"),
                              "wildcard_rank": t.get("wildCardRank"), "clinch": t.get("clinchIndicator")})
    js = _get(f"{STATSAPI}/schedule", {"sportId": 1, "season": cfg.season, "gameType": "F,D,L,W",
                                        "hydrate": "probablePitcher,seriesStatus"}).json()
    games = []
    for d in js.get("dates", []):
        for g in d["games"]:
            def side(s):
                t = g["teams"][s]
                pp = t.get("probablePitcher") or {}
                return {"team_id": int(t["team"]["id"]), "team": ID2ABBR.get(int(t["team"]["id"])),
                        "name": t["team"].get("name"), "score": t.get("score"), "probable": pp.get("id"),
                        "probable_name": pp.get("fullName")}
            games.append({"game_pk": int(g["gamePk"]), "date": d["date"], "game_date": g.get("gameDate"),
                          "game_type": g.get("gameType"), "description": g.get("description") or "",
                          "series": g.get("seriesDescription"), "game_number": g.get("seriesGameNumber"),
                          "games_in_series": g.get("gamesInSeries"), "status": g["status"].get("detailedState"),
                          "abstract": g["status"].get("abstractGameState"), "if_necessary": g.get("ifNecessary"),
                          "venue": (g.get("venue") or {}).get("name"), "home": side("home"), "away": side("away"),
                          "series_status": (g.get("seriesStatus") or {}).get("result")})
    path = DATA_RAW / f"bracket_{cfg.season}.json"
    path.write_text(json.dumps({"pulled": datetime.now().isoformat(timespec="minutes"), "teams": teams,
                                "games": games}, indent=1), encoding="utf-8")
    if verbose:
        print(f"  bracket: {len(teams)} teams, {len(games)} postseason games "
              f"({sum(g['abstract'] == 'Final' for g in games)} final)")
    return path


def load_bracket(cfg: Config) -> dict | None:
    path = DATA_RAW / f"bracket_{cfg.season}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pull_espn_upcoming(dates, verbose: bool = True) -> dict:
    """Current lines for games that have not started yet (not cached: they move until first pitch)."""
    if os.environ.get("ALCS_OFFLINE"):
        return {}
    out = {}
    for d in sorted({pd.Timestamp(x).strftime("%Y%m%d") for x in dates}):
        try:
            sb = _get(f"{ESPN}/scoreboard", {"dates": d, "limit": 50}).json()
        except Exception as e:
            print(f"  warning: ESPN scoreboard {d} failed ({e})")
            continue
        for ev in sb.get("events", []):
            comp = ev["competitions"][0]
            if comp.get("status", {}).get("type", {}).get("state") != "pre":
                continue
            teams = {c["homeAway"]: c["team"]["abbreviation"] for c in comp["competitors"]}
            try:
                s = _get(f"{ESPN}/summary", {"event": ev["id"]}).json()
                lines = _parse_pick(s.get("pickcenter") or [])
            except Exception:
                lines = {}
            if lines.get("p_home_close"):
                out[str(ev["id"])] = {"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "start": ev.get("date"),
                                      "home": ESPN_ABBR.get(teams.get("home"), teams.get("home")),
                                      "away": ESPN_ABBR.get(teams.get("away"), teams.get("away")), "lines": lines}
    if verbose:
        print(f"  ESPN upcoming: lines for {len(out)} games")
    return out
