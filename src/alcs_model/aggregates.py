"""Descriptive aggregates for the dashboard: team and player pitch-type tables, zones, splits,
records, conditions, defense and park factors. Raw (unshrunk) values, as reported by Savant.

Metric arrays use the field order in FIELDS so the dashboard can index them compactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import PITCH_GROUP

FIELDS = ["n", "rv100", "whiff", "xwoba", "pa", "k", "bb", "hh", "velo", "woba", "swing"]
ZONES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
MAIN_PITCHES = [p for p in PITCH_GROUP if p not in ("CS", "FO", "SC")]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df
    pae = d["pa_end"]
    den = d["woba_denom"].fillna(0).where(pae, 0)
    xw = d["estimated_woba_using_speedangle"].fillna(d["woba_value"])
    ev = d["events"].fillna("")
    return d.assign(
        _one=1, _sw=d["is_swing"].astype(int), _wh=d["is_whiff"].astype(int), _pa=pae.astype(int),
        _den=den, _wn=d["woba_value"].fillna(0).where(den > 0, 0), _xn=xw.fillna(0).where(den > 0, 0),
        _k=d["outcome"].eq("K").fillna(False).astype(int),
        _bb=ev.isin(["walk", "intent_walk"]).astype(int),
        _bip=(d["is_bip"] & pae).astype(int),
        _hh=(d["is_bip"] & pae & (d["launch_speed"] >= 95)).astype(int),
        _v=d["release_speed"].fillna(0), _vn=d["release_speed"].notna().astype(int))


def table(df: pd.DataFrame, key, persp: str = "b", fields: list[str] | None = None,
          min_n: int = 1) -> dict:
    """Group `df` by `key` (column name, list of names, or a Series) and return {key: [metrics]}."""
    if len(df) == 0:
        return {}
    d = df if "_one" in df.columns else _prep(df)
    g = d.groupby(key, observed=True)
    s = g[["_one", "_sw", "_wh", "_pa", "_den", "_wn", "_xn", "_k", "_bb", "_bip", "_hh", "_v", "_vn"]].sum()
    rv = g["delta_run_exp"].sum()
    sign = -1 if persp == "p" else 1
    out = {}
    for k_, r in s.iterrows():
        if r["_one"] < min_n:
            continue
        m = {
            "n": int(r["_one"]), "rv100": round(sign * rv[k_] / r["_one"] * 100, 3),
            "whiff": round(r["_wh"] / r["_sw"], 3) if r["_sw"] else None,
            "xwoba": round(r["_xn"] / r["_den"], 3) if r["_den"] else None,
            "pa": int(r["_pa"]), "k": round(r["_k"] / r["_pa"], 3) if r["_pa"] else None,
            "bb": round(r["_bb"] / r["_pa"], 3) if r["_pa"] else None,
            "hh": round(r["_hh"] / r["_bip"], 3) if r["_bip"] else None,
            "velo": round(r["_v"] / r["_vn"], 1) if r["_vn"] else None,
            "woba": round(r["_wn"] / r["_den"], 3) if r["_den"] else None,
            "swing": round(r["_sw"] / r["_one"], 3),
            "rv": round(sign * rv[k_], 2),
        }
        kk = k_ if not isinstance(k_, tuple) else "_".join(str(x) for x in k_)
        if isinstance(kk, float) and kk.is_integer():
            kk = int(kk)
        out[str(kk)] = [m[f] for f in (fields or FIELDS)]
    return out


def team_tables(p: pd.DataFrame, team: str, active_pitchers: list[int]) -> dict:
    B = _prep(p[p["bat_team"].eq(team)])
    P = _prep(p[p["fld_team"].eq(team)])
    starts = set(P.groupby("game_pk").head(1)[["game_pk", "pitcher"]].itertuples(index=False, name=None))
    P = P.assign(_role=[("SP" if (g, pt) in starts else "RP") for g, pt in zip(P["game_pk"], P["pitcher"])])
    Pa = P[P["pitcher"].isin(active_pitchers)]
    zf = ["n", "rv100", "whiff", "xwoba"]
    month = B["game_date"].dt.strftime("%Y-%m")
    t = {
        "batAll": table(B, B["_one"].map(lambda _: "all")),
        "batPT": table(B, "pitch_type"), "batPTvsL": table(B[B["p_throws"].eq("L")], "pitch_type"),
        "batPTvsR": table(B[B["p_throws"].eq("R")], "pitch_type"),
        "batHand": table(B, "p_throws"),
        "batZoneGrp": table(B[B["zone"].notna() & B["pgroup"].notna()], [B["pgroup"], B["zone"].astype("Int64")], fields=zf),
        "batZone": table(B[B["zone"].notna()], B["zone"].astype("Int64"), fields=zf + ["swing"]),
        "batCount": table(B, "count_state"), "batVelo": table(B[B["velo_band"].notna()], "velo_band"),
        "batMonth": table(B, month, fields=["pa", "woba", "xwoba", "k", "bb"]),
        "batHomeAway": table(B, np.where(B["home_team"].eq(team), "home", "away"), fields=["pa", "woba", "xwoba", "k", "bb", "hh"]),
        "pitAll": table(P, P["_one"].map(lambda _: "all"), "p"),
        "pitRole": table(P, "_role", "p"),
        "pitLateClose": table(P[P["_role"].eq("RP") & (P["inning"] >= 7) & (P["score_diff_bat"].abs() <= 2)],
                              P["_one"].map(lambda _: "lc"), "p"),
        "pitPT": table(P, "pitch_type", "p"), "pitPTact": table(Pa, "pitch_type", "p"),
        "pitPTactVsL": table(Pa[Pa["stand"].eq("L")], "pitch_type", "p"),
        "pitPTactVsR": table(Pa[Pa["stand"].eq("R")], "pitch_type", "p"),
        "pitHand": table(P, "stand", "p"),
        "pitZoneGrp": table(P[P["zone"].notna() & P["pgroup"].notna()], [P["pgroup"], P["zone"].astype("Int64")], "p", zf),
        "pitMonth": table(P, P["game_date"].dt.strftime("%Y-%m"), "p", ["pa", "woba", "xwoba", "k", "bb"]),
        "pitHomeAway": table(P, np.where(P["home_team"].eq(team), "home", "away"), "p", ["pa", "woba", "xwoba", "k", "bb", "hh"]),
        "pitTTO": table(P[P["_role"].eq("SP") & P["n_thruorder_pitcher"].notna()], P["n_thruorder_pitcher"].astype("Int64"), "p",
                        ["pa", "woba", "xwoba", "k", "bb"]),
    }
    # keep only common pitch types in pitch tables
    for k in [k for k in t if k.startswith(("batPT", "pitPT"))]:
        t[k] = {pt: v for pt, v in t[k].items() if pt in MAIN_PITCHES and v[0] >= 20}
    return t


def hitter_cards(p: pd.DataFrame, team: str, hitters: list[int], names: dict[int, str], since: str) -> list[dict]:
    B = _prep(p[p["bat_team"].eq(team)])
    out = []
    for h in hitters:
        r = B[B["batter"].eq(h)]
        if r.empty:
            continue
        stands = sorted(set(r["stand"].dropna().astype(str)))
        a = table(r, r["_one"].map(lambda _: "a"), fields=["pa", "woba", "xwoba", "k", "bb", "hh", "rv", "whiff"])["a"]
        pt = table(r, "pitch_type", fields=["n", "rv100", "whiff", "xwoba", "pa"])
        out.append({
            "id": str(h), "name": names.get(h, str(h)), "bats": "S" if len(stands) > 1 else (stands[0] if stands else "R"),
            "hand": table(r, "p_throws", fields=["pa", "woba", "xwoba", "k", "bb", "hh"]),
            "all": a,
            "pt": {k: v for k, v in pt.items() if k in MAIN_PITCHES},
            "grpH": table(r[r["pgroup"].notna()], [r["pgroup"], r["p_throws"]], fields=["n", "rv100", "whiff", "xwoba", "pa"]),
            "zone": table(r[r["zone"].notna()], r["zone"].astype("Int64"), fields=["n", "rv100", "whiff", "xwoba"]),
            "velo": table(fb := r[r["pgroup"].eq("FB") & r["release_speed"].notna()],
                          np.where(fb["release_speed"] >= 95, "95+", "<95"), fields=["n", "rv100", "whiff", "xwoba"]),
            "sep": (table(r[r["game_date"] >= pd.Timestamp(since)], r["_one"].map(lambda _: "s"),
                          fields=["pa", "woba", "xwoba", "k"]) or {"s": None}).get("s"),
            "k2": (table(r[r["strikes"].eq(2)], r["_one"].map(lambda _: "s"), fields=["n", "whiff", "pa", "k", "xwoba"])
                   or {"s": None}).get("s"),
        })
    return out


def _zone_pct(rows: pd.DataFrame) -> list:
    z = rows["zone"].dropna().astype(int)
    n = len(z)
    if not n:
        return [0] + [0.0] * 13
    c = z.value_counts()
    return [int(n)] + [round(float(c.get(k, 0)) / n * 100, 1) for k in ZONES]


def pitcher_cards(p: pd.DataFrame, team: str, pitchers: list[int], names: dict[int, str], since: str) -> list[dict]:
    P = _prep(p[p["fld_team"].eq(team)])
    starts = P.groupby("game_pk").head(1)
    out = []
    for pid in pitchers:
        r = P[P["pitcher"].eq(pid)]
        if r.empty:
            continue
        n = len(r)
        gs = int(starts["pitcher"].eq(pid).sum())
        st_games = set(starts.loc[starts["pitcher"].eq(pid), "game_pk"])
        nL = int(r["stand"].eq("L").sum())
        nR = n - nL
        two = r[r["strikes"].eq(2)]
        pt = table(r, "pitch_type", "p", ["n", "velo", "whiff", "rv100", "xwoba"])
        ars = {}
        for k, v in pt.items():
            if v[0] / n < 0.02 or k not in MAIN_PITCHES:
                continue
            cL = int((r["stand"].eq("L") & r["pitch_type"].eq(k)).sum())
            cR = int((r["stand"].eq("R") & r["pitch_type"].eq(k)).sum())
            c2 = int(two["pitch_type"].eq(k).sum())
            ars[k] = [v[0], round(v[0] / n * 100, 1), round(cL / nL * 100, 1) if nL else 0, round(cR / nR * 100, 1) if nR else 0,
                      v[1], v[2], v[3], v[4], round(c2 / len(two) * 100, 1) if len(two) else 0]
        sp = r[r["game_pk"].isin(st_games)]
        lc = r[~r["game_pk"].isin(st_games) & (r["inning"] >= 7) & (r["score_diff_bat"].abs() <= 2)]
        late = r[r["game_date"] >= pd.Timestamp(since)]
        out.append({
            "id": str(pid), "name": names.get(pid, str(pid)), "th": str(r["p_throws"].iloc[0]),
            "g": int(r["game_pk"].nunique()), "gs": gs, "n": n,
            "all": table(r, r["_one"].map(lambda _: "a"), "p", ["pa", "woba", "xwoba", "k", "bb", "hh", "rv", "rv100", "whiff"])["a"],
            "hand": table(r, "stand", "p", ["pa", "woba", "xwoba", "k", "bb"]),
            "ars": ars, "zAll": _zone_pct(r), "zFB": _zone_pct(r[r["pgroup"].eq("FB")]),
            "zBO": _zone_pct(r[r["pgroup"].isin(["BR", "OS"])]),
            "tto": table(sp[sp["n_thruorder_pitcher"].notna()], sp["n_thruorder_pitcher"].astype("Int64"), "p", ["pa", "xwoba", "k"]) if gs >= 5 else None,
            "lc": (table(lc, lc["_one"].map(lambda _: "a"), "p", ["pa", "xwoba", "k", "rv"]) or {"a": None}).get("a"),
            "sep": (table(late, late["_one"].map(lambda _: "a"), "p", ["pa", "xwoba", "k", "bb", "velo"]) or {"a": None}).get("a"),
        })
    return out


def game_results(p: pd.DataFrame, team: str) -> pd.DataFrame:
    d = p[(p["home_team"].eq(team) | p["away_team"].eq(team))]
    g = d.groupby("game_pk").agg(date=("game_date", "first"), home=("home_team", "first"), away=("away_team", "first"),
                                 game_type=("game_type", "first"),
                                 hs=("post_home_score", "max"), as_=("post_away_score", "max"))
    g["is_home"] = g["home"].eq(team)
    g["opp"] = np.where(g["is_home"], g["away"], g["home"])
    g["rf"] = np.where(g["is_home"], g["hs"], g["as_"])
    g["ra"] = np.where(g["is_home"], g["as_"], g["hs"])
    return g.reset_index().sort_values("date")


def records(p: pd.DataFrame, cfg) -> dict:
    out = {}
    a, b = cfg.teams
    for t in cfg.teams:
        G = game_results(p, t)
        R = G[G["game_type"].astype(str).eq("R")] if "game_type" in G else G

        def rec(s):
            return {"g": int(len(s)), "w": int((s["rf"] > s["ra"]).sum()), "l": int((s["rf"] < s["ra"]).sum()),
                    "rf": int(s["rf"].sum()), "ra": int(s["ra"].sum())}
        last_month = R["date"].max() - pd.Timedelta(days=30) if len(R) else None
        out[t] = {"all": rec(R), "home": rec(R[R["is_home"]]), "away": rec(R[~R["is_home"]]),
                  "sep": rec(R[R["date"] >= last_month]) if last_month is not None else rec(R.iloc[:0]),
                  "onerun": rec(R[(R["rf"] - R["ra"]).abs() == 1]),
                  "post": rec(G[~G["game_type"].astype(str).eq("R")])}
    G = game_results(p, a)
    h2h = G[G["opp"].eq(b) & G["game_type"].astype(str).eq("R")]
    out["h2h"] = [[str(r.date.date()), r.home, int(r.rf), int(r.ra), str(r.game_pk)] for r in h2h.itertuples()]
    out["h2h_note"] = f"rows: date, home team, {a} runs, {b} runs, game_pk"
    gp = set(h2h["game_pk"])
    H = _prep(p[p["game_pk"].isin(gp)])
    out["h2h_batting"] = {t: (table(H[H["bat_team"].eq(t)], H.loc[H["bat_team"].eq(t), "_one"].map(lambda _: "a"),
                                    fields=["pa", "woba", "xwoba", "k", "bb", "hh"]) or {"a": None}).get("a") for t in cfg.teams}
    return out


def conditions(p: pd.DataFrame, sched: pd.DataFrame | None, cfg) -> dict:
    out = {}
    if sched is None or sched.empty:
        return out
    s = sched.drop_duplicates("game_pk").set_index("game_pk")
    indoor = s["condition"].astype(str).str.contains("Dome|Roof Closed", case=False, regex=True)

    def tb(gpk):
        if gpk not in s.index:
            return None
        if indoor.get(gpk, False):
            return "indoor"
        t = s.at[gpk, "temp"]
        if pd.isna(t):
            return None
        return "<60" if t < 60 else ("60-74" if t < 75 else "75+")
    for t in cfg.teams:
        G = game_results(p, t)
        G = G[G["game_type"].astype(str).eq("R")]
        G["dn"] = G["game_pk"].map(s["day_night"]) if "day_night" in s else None
        G["tb"] = G["game_pk"].map(tb)

        def rec(x):
            return [int(len(x)), int((x["rf"] > x["ra"]).sum()), int((x["rf"] < x["ra"]).sum()), int(x["rf"].sum()), int(x["ra"].sum())]
        B = _prep(p[p["bat_team"].eq(t) & p["game_type"].astype(str).eq("R")])
        P = _prep(p[p["fld_team"].eq(t) & p["game_type"].astype(str).eq("R")])
        bdn, pdn = B["game_pk"].map(s["day_night"]), P["game_pk"].map(s["day_night"])
        btb, ptb = B["game_pk"].map(tb), P["game_pk"].map(tb)
        f6, f5 = ["pa", "woba", "xwoba", "k", "bb", "hh"], ["pa", "woba", "xwoba", "k", "hh"]
        out[t] = {
            "recDN": {"day": rec(G[G["dn"].eq("day")]), "night": rec(G[G["dn"].eq("night")]),
                      "homeDay": rec(G[G["is_home"] & G["dn"].eq("day")]), "homeNight": rec(G[G["is_home"] & G["dn"].eq("night")])},
            "recTemp": {k: rec(G[G["tb"].eq(k)]) for k in ("<60", "60-74", "75+", "indoor")},
            "batDN": table(B[bdn.notna()], bdn[bdn.notna()], fields=f6), "pitDN": table(P[pdn.notna()], pdn[pdn.notna()], "p", f6),
            "batTemp": table(B[btb.notna()], btb[btb.notna()], fields=f5), "pitTemp": table(P[ptb.notna()], ptb[ptb.notna()], "p", f5),
        }
    out["alcs_schedule"] = [[f"G{g['game']}", f"{pd.Timestamp(g['date']):%a %b} {pd.Timestamp(g['date']).day}", g["home"]]
                            for g in cfg["schedule"]]
    return out


def defense(cfg, lb: dict) -> dict:
    a, b = cfg.teams
    names = {t: cfg.team_name(t) for t in cfg.teams}
    out = {}
    oaa = lb.get("oaa_team")
    if oaa is not None:
        out["oaaTeam"] = sorted([[r.team_name, int(r.outs_above_average), int(r.outs_above_average_rhh), int(r.outs_above_average_lhh)]
                                 for r in oaa.itertuples()], key=lambda x: -x[1])
    oaap = lb.get("oaa_player")
    if oaap is not None:
        sel = oaap[oaap["display_team_name"].isin(names.values())]
        out["oaaPlayer"] = sorted([[r["display_team_name"], r["last_name, first_name"], r["primary_pos_formatted"],
                                    int(r["fielding_runs_prevented"]), int(r["outs_above_average"])]
                                   for _, r in sel.iterrows()], key=lambda x: -x[4])
    sp = lb.get("sprint")
    if sp is not None:
        lg = sp["sprint_speed"].astype(float)
        out["sprintLgAvg"] = round(float(lg.mean()), 2)
        sel = sp[sp["team"].isin(cfg.teams)]
        out["sprint"] = [[r["team"], r["last_name, first_name"], r["position"], float(r["sprint_speed"]),
                          float(r["hp_to_1b"]) if pd.notna(r["hp_to_1b"]) else None, int((lg < float(r["sprint_speed"])).mean() * 100)]
                         for _, r in sel.sort_values("sprint_speed", ascending=False).iterrows()]
    br = lb.get("baserunning")
    if br is not None:
        tot = br.groupby("team_name")["runner_runs_tot"].sum().sort_values(ascending=False)
        out["brTeam"] = {t: [round(float(tot.get(t, 0)), 1), int(list(tot.index).index(t) + 1) if t in tot.index else None] for t in cfg.teams}
        sel = br[br["team_name"].isin(cfg.teams)].sort_values("runner_runs_tot", ascending=False)
        out["brPlayers"] = [[r["team_name"], r["entity_name"], round(float(r["runner_runs_tot"]), 1), round(float(r["runner_runs_XB"]), 1),
                             round(float(r["runner_runs_SBX"]), 1)] for _, r in sel.iterrows()]
    fr = lb.get("framing")
    if fr is not None:
        out["catcherAll"] = fr[["name", "pitches", "rv_tot", "pct_tot"]].round(3).values.tolist()
    return out


def park(cfg, pf: pd.DataFrame | None) -> dict:
    if pf is None or pf.empty:
        return {}
    fields = ["index_woba", "index_runs", "index_hr", "index_2b", "index_3b", "index_1b", "index_bb", "index_so",
              "index_hardhit", "index_wobacon", "n_pa", "year_range"]
    out = {"fields": ["wOBA", "R", "HR", "2B", "3B", "1B", "BB", "SO", "HardHit", "wOBAcon", "PA", "years"]}
    venues = [cfg.venue(t) for t in cfg.teams]
    keymap = {(1, "All"): "1yr", (3, "All"): "3yr", (3, "L"): "3yr_LHH", (3, "R"): "3yr_RHH", (1, "L"): "1yr_LHH", (1, "R"): "1yr_RHH"}
    for (yrs, side), name in keymap.items():
        sub = pf[(pf["rolling_years"] == yrs) & (pf["bat_side_key"].astype(str) == side)]
        out[name] = {v: [(int(x) if str(x).isdigit() else x) for x in sub[sub["venue_name"].eq(v)][fields].iloc[0].tolist()]
                     for v in venues if (sub["venue_name"] == v).any()}
    return out


def park_multipliers(cfg, pf: pd.DataFrame | None) -> dict:
    """(venue, bat side) -> multipliers on [K, BB, 1B, 2B, 3B, HR, OUT] from 3-year park factors."""
    out = {}
    if pf is None or pf.empty:
        return out
    for v in [cfg.venue(t) for t in cfg.teams]:
        for side in ("L", "R"):
            sub = pf[(pf["rolling_years"] == 3) & (pf["bat_side_key"].astype(str) == side) & pf["venue_name"].eq(v)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            f = lambda c: float(r[c]) / 100 if pd.notna(r[c]) else 1.0  # noqa: E731
            out[(v, side)] = np.array([f("index_so"), f("index_bb"), f("index_1b"), f("index_2b"), f("index_3b"), f("index_hr"), 1.0])
    return out


def park_run_index(cfg, pf: pd.DataFrame | None, venues: list[str]) -> float:
    """Average 3-year runs park factor over the series venues (1.0 = neutral)."""
    if pf is None or pf.empty:
        return 1.0
    sub = pf[(pf["rolling_years"] == 3) & (pf["bat_side_key"].astype(str) == "All")]
    vals = []
    for v in venues:
        r = sub[sub["venue_name"].eq(v)]
        vals.append(float(r["index_runs"].iloc[0]) / 100 if len(r) and pd.notna(r["index_runs"].iloc[0]) else 1.0)
    return float(np.mean(vals)) if vals else 1.0
