"""Turn raw Statcast rows into analysis-ready pitch and plate-appearance tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

PITCH_GROUP = {"FF": "FB", "SI": "FB", "FC": "FB", "SL": "BR", "ST": "BR", "CU": "BR", "KC": "BR",
               "SV": "BR", "CS": "BR", "CH": "OS", "FS": "OS", "FO": "OS", "SC": "OS"}
PITCH_NAME = {"FF": "4-Seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper", "CU": "Curveball",
              "KC": "Knuckle-curve", "SV": "Slurve", "CS": "Slow curve", "CH": "Changeup", "FS": "Splitter",
              "FO": "Forkball", "SC": "Screwball"}
SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt",
          "missed_bunt", "bunt_foul_tip"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
OUTCOMES = ["K", "BB", "1B", "2B", "3B", "HR", "OUT"]
EVENT_MAP = {
    "strikeout": "K", "strikeout_double_play": "K",
    "walk": "BB", "intent_walk": "BB", "hit_by_pitch": "BB", "catcher_interf": "BB",
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
    "field_out": "OUT", "force_out": "OUT", "grounded_into_double_play": "OUT", "double_play": "OUT",
    "triple_play": "OUT", "sac_fly": "OUT", "sac_bunt": "OUT", "fielders_choice": "OUT",
    "fielders_choice_out": "OUT", "sac_fly_double_play": "OUT", "sac_bunt_double_play": "OUT",
    "field_error": "1B",  # reached on error: treated like a single for run scoring
}
# wOBA weights (approximate, stable across recent seasons) used only for expected-wOBA summaries
WOBA_W = {"K": 0.0, "BB": 0.70, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10, "OUT": 0.0}

NUMERIC = ["release_speed", "zone", "balls", "strikes", "outs_when_up", "inning", "launch_speed",
           "launch_angle", "estimated_woba_using_speedangle", "woba_value", "woba_denom", "delta_run_exp",
           "delta_home_win_exp", "home_win_exp", "bat_score", "fld_score", "home_score", "away_score",
           "post_home_score", "post_away_score", "post_bat_score", "n_thruorder_pitcher",
           "pitcher_days_since_prev_game", "at_bat_number", "pitch_number", "pfx_x", "pfx_z",
           "release_spin_rate", "release_extension", "arm_angle", "plate_x", "plate_z", "batter", "pitcher",
           "game_pk", "on_1b", "on_2b", "on_3b"]


def prepare_pitches(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean types and add derived columns. Returns a new frame (one row per pitch)."""
    df = raw.copy()
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("pitch_type", "description", "events", "stand", "p_throws", "home_team", "away_team",
              "inning_topbot", "game_type", "player_name"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["pitch_type"].notna() & ~df["pitch_type"].isin(["PO", "IN", "UN", "AB"])].copy()
    top = df["inning_topbot"].eq("Top")
    df["bat_team"] = np.where(top, df["away_team"], df["home_team"])
    df["fld_team"] = np.where(top, df["home_team"], df["away_team"])
    df["pgroup"] = df["pitch_type"].map(PITCH_GROUP)
    desc = df["description"].fillna("")
    df["is_swing"] = desc.isin(SWINGS)
    df["is_whiff"] = desc.isin(WHIFFS)
    ev = df["events"].fillna("")
    df["outcome"] = ev.map(EVENT_MAP)
    df["pa_end"] = df["outcome"].notna()
    df["is_bip"] = desc.eq("hit_into_play")
    df["base_state"] = (df["on_1b"].notna().astype(int) + 2 * df["on_2b"].notna().astype(int)
                        + 4 * df["on_3b"].notna().astype(int))
    df["score_diff_bat"] = df["bat_score"] - df["fld_score"]
    df["delta_run_exp"] = df["delta_run_exp"].fillna(0.0)
    df["xrv"] = expected_run_value(df)
    df["count_state"] = np.select(
        [df["strikes"].eq(2), df["balls"] > df["strikes"], df["balls"] < df["strikes"]],
        ["2K", "ahead", "behind"], default="even")
    df["velo_band"] = np.where(df["pgroup"].ne("FB") | df["release_speed"].isna(), None,
                               pd.cut(df["release_speed"], [0, 93, 95, 97, 200], right=False,
                                      labels=["<93", "93-95", "95-97", "97+"]).astype("string"))
    return df.reset_index(drop=True)


def expected_run_value(df: pd.DataFrame) -> pd.Series:
    """Run value with batted-ball luck removed.

    For balls in play, replace the actual run value with a linear map from xwOBA to run value,
    fit on the same data (delta_run_exp ~ a + b * woba_value on balls in play). Non-contact pitches keep
    their actual run value. This strips most fielding and sequencing noise from small samples.
    """
    bip = df["description"].eq("hit_into_play") & df["woba_value"].notna()
    out = df["delta_run_exp"].astype(float).copy()
    if bip.sum() < 50:
        return out
    x = df.loc[bip, "woba_value"].astype(float).to_numpy()
    y = df.loc[bip, "delta_run_exp"].astype(float).to_numpy()
    b, a = np.polyfit(x, y, 1)
    xw = df.loc[bip, "estimated_woba_using_speedangle"].astype(float)
    xw = xw.fillna(df.loc[bip, "woba_value"].astype(float))
    pred = a + b * xw.to_numpy()
    out.loc[bip] = pred - (pred.mean() - y.mean())  # keep the BIP total equal to actual run value
    return out


def recency_weights(dates: pd.Series, half_life_days: float, as_of: pd.Timestamp | None = None) -> np.ndarray:
    if not half_life_days:
        return np.ones(len(dates))
    as_of = as_of if as_of is not None else dates.max()
    age = (as_of - dates).dt.days.clip(lower=0).to_numpy()
    w = 0.5 ** (age / half_life_days)
    # Normalize to mean 1 so recency tilts each player's sample toward recent games without
    # shrinking the effective sample size (which would double-count regression to the mean).
    return w / w.mean() if len(w) else w


def plate_appearances(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per completed plate appearance with the pre-PA game state."""
    first = pitches.groupby(["game_pk", "at_bat_number"], sort=False).head(1)
    state_cols = ["game_pk", "at_bat_number", "inning", "inning_topbot", "outs_when_up", "base_state",
                  "score_diff_bat", "home_win_exp"]
    pre = first[state_cols].rename(columns={"home_win_exp": "pre_home_we"})
    wpa = pitches.groupby(["game_pk", "at_bat_number"], sort=False)["delta_home_win_exp"].sum().rename("dwe")
    last = pitches[pitches["pa_end"]].drop_duplicates(["game_pk", "at_bat_number"], keep="last")
    keep = ["game_pk", "at_bat_number", "game_date", "game_type", "batter", "pitcher", "stand", "p_throws",
            "bat_team", "fld_team", "home_team", "away_team", "outcome", "woba_value", "woba_denom",
            "estimated_woba_using_speedangle", "n_thruorder_pitcher", "events"]
    keep = [c for c in keep if c in last.columns]
    pa = last[keep].merge(pre.drop(columns=[]), on=["game_pk", "at_bat_number"], how="left",
                          suffixes=("", "_pre"))
    pa = pa.merge(wpa, left_on=["game_pk", "at_bat_number"], right_index=True, how="left")
    pa["xwoba"] = pa["estimated_woba_using_speedangle"].fillna(pa["woba_value"])
    return pa.reset_index(drop=True)


def leverage_table(pa: pd.DataFrame, k: float = 30.0) -> pd.DataFrame:
    """Empirical leverage index by game state.

    LI = average absolute win-probability swing of a PA in that state, divided by the average swing
    across all PAs. Sparse states are smoothed toward coarser states (inning/half/score, then score).
    """
    d = pa.dropna(subset=["dwe"]).copy()
    d["absdwe"] = d["dwe"].abs()
    base = d["absdwe"].mean()
    d["inn"] = d["inning"].clip(upper=9).astype(int)
    home_bat = d["inning_topbot"].eq("Bot")
    d["hsd"] = np.where(home_bat, d["score_diff_bat"], -d["score_diff_bat"]).clip(-4, 4)
    fine = ["inn", "inning_topbot", "outs_when_up", "base_state", "hsd"]
    mid = ["inn", "inning_topbot", "hsd"]
    coarse = ["hsd"]
    g_c = d.groupby(coarse)["absdwe"].agg(["mean", "size"]).rename(columns={"mean": "m_c", "size": "n_c"})
    g_m = d.groupby(mid)["absdwe"].agg(["mean", "size"]).rename(columns={"mean": "m_m", "size": "n_m"})
    g_f = d.groupby(fine)["absdwe"].agg(["mean", "size"]).rename(columns={"mean": "m_f", "size": "n_f"})
    t = g_f.reset_index().merge(g_m.reset_index(), on=mid).merge(g_c.reset_index(), on=coarse)
    t["m_mid"] = (t["m_m"] * t["n_m"] + t["m_c"] * k) / (t["n_m"] + k)
    t["m"] = (t["m_f"] * t["n_f"] + t["m_mid"] * k) / (t["n_f"] + k)
    t["li"] = t["m"] / base
    return t[fine + ["li", "n_f"]]


def add_leverage(pa: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    d = pa.copy()
    d["inn"] = d["inning"].clip(upper=9).astype(int)
    home_bat = d["inning_topbot"].eq("Bot")
    d["hsd"] = np.where(home_bat, d["score_diff_bat"], -d["score_diff_bat"]).clip(-4, 4)
    d = d.merge(table, on=["inn", "inning_topbot", "outs_when_up", "base_state", "hsd"], how="left")
    d["li"] = d["li"].fillna(1.0)
    return d.drop(columns=["inn", "hsd", "n_f"], errors="ignore")


def game_starters(pitches: pd.DataFrame) -> pd.DataFrame:
    """First pitcher used by each fielding team in each game."""
    first = pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"]).groupby(
        ["game_pk", "fld_team"], sort=False).head(1)
    return first[["game_pk", "fld_team", "pitcher"]].rename(columns={"pitcher": "starter"})


def active_players(pitches: pd.DataFrame, team: str, since: str, min_pitches: int = 40,
                   min_pa: int = 15, min_season_pa: int = 100) -> tuple[list[int], list[int]]:
    """Pitchers and hitters who were active late in the season (a stand-in for the playoff roster)."""
    late = pitches[pitches["game_date"] >= pd.Timestamp(since)]
    pit = late[late["fld_team"].eq(team)].groupby("pitcher").size()
    pitchers = pit[pit >= min_pitches].index.astype(int).tolist()
    pa_all = pitches[pitches["bat_team"].eq(team) & pitches["pa_end"]].groupby("batter").size()
    pa_late = late[late["bat_team"].eq(team) & late["pa_end"]].groupby("batter").size()
    hitters = [int(b) for b, n in pa_late.items() if n >= min_pa and pa_all.get(b, 0) >= min_season_pa]
    return pitchers, hitters


def player_names(pitches: pd.DataFrame) -> dict[int, str]:
    """MLBAM id -> 'Last, First'. Savant's player_name is the pitcher for pitch-level pulls."""
    names = {}
    if "player_name" in pitches.columns:
        for pid, nm in pitches[["pitcher", "player_name"]].dropna().drop_duplicates("pitcher").itertuples(index=False):
            names[int(pid)] = str(nm)
    return names
