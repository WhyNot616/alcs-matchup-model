"""Synthetic Statcast-shaped data for tests. Not realistic baseball, just the right columns and
internally consistent game states, so every part of the pipeline can run without network access."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from alcs_model.simulate import _advance

OUT_RV = {"K": -0.27, "BB": 0.32, "1B": 0.47, "2B": 0.77, "3B": 1.05, "HR": 1.40, "OUT": -0.26}
OUT_WOBA = {"K": 0.0, "BB": 0.70, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10, "OUT": 0.0}
EVENTS = {"K": "strikeout", "BB": "walk", "1B": "single", "2B": "double", "3B": "triple", "HR": "home_run", "OUT": "field_out"}
OUTS = ["K", "BB", "1B", "2B", "3B", "HR", "OUT"]
BASE = np.array([0.225, 0.085, 0.14, 0.045, 0.004, 0.03, 0.471])
ARSENALS = [["FF", "SL", "CH"], ["SI", "FC", "ST"], ["FF", "CU", "CH", "SI"], ["FF", "SL"], ["SI", "CH", "FC", "CU"]]

CWS_HIT = [803011, 805367, 808959, 678246, 695657, 643217, 671976, 695731, 691019, 545341]
TB_HIT = [802415, 650490, 691406, 666018, 676356, 689414, 680700, 656775, 670764, 668723, 663743]
CWS_SP = [696146, 641743, 680732, 663436, 702273]
TB_SP = [656876, 642547, 607259, 643377, 693855]


def make_league(seed: int = 7, n_days: int = 90, start: date = date(2026, 4, 1), teams=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = list(teams or ["CWS", "TB", "CLE", "NYY", "HOU", "BOS"])
    hitters, pitchers, sp, rp = {}, {}, {}, {}
    nid = 900000
    for t in teams:
        if t == "CWS":
            hs = CWS_HIT
        elif t == "TB":
            hs = TB_HIT
        else:
            hs = list(range(nid, nid + 10))
            nid += 10
        hitters[t] = hs
        s = CWS_SP if t == "CWS" else TB_SP if t == "TB" else list(range(nid, nid + 5))
        nid += 5 if t not in ("CWS", "TB") else 0
        r = list(range(nid, nid + 7))
        nid += 7
        sp[t], rp[t] = s, r
        pitchers[t] = s + r
    bat_skill = {h: rng.normal(0, 0.25) for t in teams for h in hitters[t]}
    bat_hand = {h: rng.choice(["L", "R", "S"], p=[0.4, 0.5, 0.1]) for t in teams for h in hitters[t]}
    pit_skill = {p: rng.normal(0, 0.25) for t in teams for p in pitchers[t]}
    pit_hand = {p: rng.choice(["L", "R"], p=[0.3, 0.7]) for t in teams for p in pitchers[t]}
    arsenal = {p: ARSENALS[i % len(ARSENALS)] for i, p in enumerate(pid for t in teams for pid in pitchers[t])}
    velo = {p: rng.normal(94, 2) for t in teams for p in pitchers[t]}
    rows = []
    gpk = 700000
    last_pitched: dict[int, date] = {}
    rot_i = {t: 0 for t in teams}
    for d in range(n_days):
        day = start + timedelta(days=d)
        order = rng.permutation(teams)
        for gi in range(0, len(order), 2):
            home, away = order[gi], order[gi + 1]
            gpk += 1
            rows.extend(_game(rng, gpk, day, home, away, hitters, sp, rp, rot_i, bat_skill, bat_hand, pit_skill,
                              pit_hand, arsenal, velo, last_pitched))
    df = pd.DataFrame(rows)
    return df


def _game(rng, gpk, day, home, away, hitters, sp, rp, rot_i, bat_skill, bat_hand, pit_skill, pit_hand, arsenal,
          velo, last_pitched):
    rows = []
    score = {home: 0, away: 0}
    idx = {home: 0, away: 0}
    cur, bf, maxbf, used = {}, {}, {}, {}
    for t in (home, away):
        cur[t] = sp[t][rot_i[t] % 5]
        rot_i[t] += 1
        bf[t] = 0
        maxbf[t] = int(rng.normal(23, 4))
        used[t] = {cur[t]}
    lineup = {t: [int(x) for x in rng.choice(hitters[t], 9, replace=False)] for t in (home, away)}
    faced: dict[tuple[int, int], int] = {}
    ab = 0
    we = 0.5
    inning = 1
    while True:
        for half, bat in (("Top", away), ("Bot", home)):
            if half == "Bot" and inning >= 9 and score[home] > score[away]:
                return rows
            fld = home if bat == away else away
            outs, bases = 0, [0, 0, 0]
            while outs < 3:
                if bf[fld] >= maxbf[fld]:
                    pool = [p for p in rp[fld] if p not in used[fld]] or rp[fld]
                    cur[fld] = int(rng.choice(pool))
                    used[fld].add(cur[fld])
                    bf[fld] = 0
                    maxbf[fld] = int(rng.integers(3, 8))
                pit = cur[fld]
                batter = lineup[bat][idx[bat] % 9]
                idx[bat] += 1
                ab += 1
                ph = pit_hand[pit]
                bh = bat_hand[batter]
                stand = ("L" if ph == "R" else "R") if bh == "S" else bh
                x = BASE * np.exp(np.array([-1, 1, 1, 1, 1, 1, -1]) * (bat_skill[batter] - pit_skill[pit]))
                if stand == ph:
                    x = x * np.array([1.08, 0.95, 0.96, 0.96, 1, 0.92, 1.0])
                x = x / x.sum()
                o = OUTS[rng.choice(7, p=x)]
                key = (gpk, pit, batter)
                faced[key] = faced.get(key, 0) + 1
                n_p = int(rng.integers(1, 7))
                pre_bases = list(bases)
                pre_outs = outs
                runs, outs2 = _advance(OUTS.index(o), bases, outs, rng)
                if outs2 >= 3 and o == "OUT":
                    runs = 0
                hs0, as0 = score[home], score[away]
                bat_score0, fld_score0 = score[bat], score[fld]
                score[bat] += runs
                new_we = 1 / (1 + np.exp(-(score[home] - score[away]) * (0.35 + 0.05 * inning)))
                dwe = new_we - we
                we = new_we
                days_since = (day - last_pitched[pit]).days if pit in last_pitched else None
                for pn in range(1, n_p + 1):
                    final = pn == n_p
                    pt = arsenal[pit][int(rng.integers(len(arsenal[pit])))]
                    if final:
                        desc = {"K": "swinging_strike", "BB": "ball"}.get(o, "hit_into_play")
                    else:
                        desc = rng.choice(["ball", "called_strike", "foul", "swinging_strike"], p=[.38, .2, .3, .12])
                    ls = rng.normal(92, 12) if final and desc == "hit_into_play" else np.nan
                    rows.append({
                        "pitch_type": pt, "game_date": day.isoformat(), "release_speed": velo[pit] - (8 if pt not in ("FF", "SI", "FC") else 0) + rng.normal(0, 1),
                        "player_name": f"P{pit}, Test", "batter": batter, "pitcher": pit,
                        "events": EVENTS[o] if final else None, "description": desc,
                        "zone": int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14])), "game_type": "R",
                        "stand": stand, "p_throws": ph, "home_team": home, "away_team": away,
                        "balls": min(pn - 1, 3), "strikes": min(pn - 1, 2), "outs_when_up": pre_outs, "inning": inning,
                        "inning_topbot": half, "on_1b": 1 if pre_bases[0] else None, "on_2b": 1 if pre_bases[1] else None,
                        "on_3b": 1 if pre_bases[2] else None, "launch_speed": ls, "launch_angle": np.nan,
                        "estimated_woba_using_speedangle": (max(0, OUT_WOBA[o] + rng.normal(0, .25)) if final and desc == "hit_into_play" else np.nan),
                        "woba_value": OUT_WOBA[o] if final else np.nan, "woba_denom": 1 if final else np.nan,
                        "delta_run_exp": (OUT_RV[o] + rng.normal(0, .05)) if final else rng.normal(0, .03),
                        "delta_home_win_exp": dwe if final else 0.0, "home_win_exp": we - dwe,
                        "bat_score": bat_score0, "fld_score": fld_score0, "home_score": hs0, "away_score": as0,
                        "post_home_score": score[home] if final else hs0, "post_away_score": score[away] if final else as0,
                        "post_bat_score": score[bat] if final else bat_score0,
                        "n_thruorder_pitcher": faced[key], "pitcher_days_since_prev_game": days_since,
                        "at_bat_number": ab, "pitch_number": pn, "game_pk": gpk,
                    })
                outs = outs2
                bf[fld] += 1
                if half == "Bot" and inning >= 9 and score[home] > score[away]:
                    for t in (home, away):
                        for p in used[t]:
                            last_pitched[p] = day
                    return rows
        if inning >= 9 and score[home] != score[away]:
            for t in (home, away):
                for p in used[t]:
                    last_pitched[p] = day
            return rows
        inning += 1
        if inning > 14:
            return rows
