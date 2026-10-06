"""Plate-appearance-level game and series simulator.

Each PA draws an outcome from pa_model.matchup_probs for the current batter and pitcher (with the
pitch-mix tilt, park and home-field adjustments), then advances runners with simple league-average
baserunning rules. Starters leave based on their own 2026 batters-faced distribution (capped for
planned openers), and relievers are picked by each team's UsageModel, with rest tracked across the
series calendar. Postseason rules: no automatic runner in extra innings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .bullpen import UsageModel
from .pa_model import GOOD, Rates, batter_side, matchup_probs

K, BB, S1, S2, S3, HR, OUT = range(7)


@dataclass
class TeamSetup:
    abbr: str
    lineups: dict[str, list[int]]           # "vs_R"/"vs_L" -> batter ids
    bats: dict[int, str]                    # batter -> L/R/S
    throws: dict[int, str]                  # pitcher -> L/R
    bullpen: list[int]
    usage: UsageModel
    starter_bf: dict[int, np.ndarray]       # pitcher -> sample of batters faced in starts
    home_park: str


@dataclass
class GamePlan:
    home: str
    venue: str
    starters: dict[str, dict]               # team -> {"starter", "opener_max_bf", "bulk"}
    day: date


@dataclass
class SimContext:
    rates: Rates
    edges: dict[tuple[int, int], float]
    lam: float
    park: dict[tuple[str, str], np.ndarray]     # (venue, bat side) -> outcome multipliers
    home_tilt: float
    teams: dict[str, TeamSetup]
    cache: dict = field(default_factory=dict)
    scoring_tilt: float = 0.0                   # global run-environment calibration (see calibrate_scoring)

    def cum_probs(self, batter: int, pitcher: int, bat_team: str, venue: str, is_home: bool) -> np.ndarray:
        key = (batter, pitcher, venue, is_home)
        c = self.cache.get(key)
        if c is not None:
            return c
        t = self.teams[bat_team]
        h = t.throws.get(pitcher) or self.teams_throw(pitcher)
        side = batter_side(t.bats.get(batter, "R"), h)
        pf = self.park.get((venue, side))
        own = self.park.get((t.home_park, side))
        mult = None
        if pf is not None:
            mult = pf / np.sqrt(own) if own is not None else pf
        p = matchup_probs(self.rates, batter, side, pitcher, h, self.edges.get((batter, pitcher), 0.0),
                          self.lam, mult)
        tilt = self.scoring_tilt + (self.home_tilt if is_home else 0.0)
        if tilt:
            p = p * np.where(GOOD, np.exp(tilt), 1.0)
            p = p / p.sum()
        c = np.cumsum(p)
        self.cache[key] = c
        return c

    def teams_throw(self, pitcher: int) -> str:
        for t in self.teams.values():
            if pitcher in t.throws:
                return t.throws[pitcher]
        return "R"


class Fatigue:
    """Tracks reliever workload by calendar day across the series."""

    def __init__(self):
        self.log: dict[int, dict[date, int]] = {}

    def add(self, pitcher: int, day: date, pitches: int):
        self.log.setdefault(pitcher, {})
        self.log[pitcher][day] = self.log[pitcher].get(day, 0) + pitches

    def available(self, pitcher: int, day: date) -> bool:
        d = self.log.get(pitcher, {})
        prev = [(day - x).days for x in d]
        y1 = any(g == 1 for g in prev)
        y2 = any(g == 2 for g in prev)
        last = d.get(next((x for x in d if (day - x).days == 1), None), 0) if y1 else 0
        if y1 and y2:
            return False          # pitched each of the last two days
        if last >= 30:
            return False          # heavy outing yesterday
        return True


def _advance(o: int, b: list[int], outs: int, rng: np.random.Generator) -> tuple[int, int]:
    """Returns (runs, outs) and mutates the base list [1B, 2B, 3B] in place."""
    runs = 0
    if o == K:
        return 0, outs + 1
    if o == BB:
        if b[0]:
            if b[1]:
                if b[2]:
                    runs += 1
                b[2] = 1
            b[1] = 1
        b[0] = 1
        return runs, outs
    if o == HR:
        runs = 1 + sum(b)
        b[:] = [0, 0, 0]
        return runs, outs
    if o == S3:
        runs = sum(b)
        b[:] = [0, 0, 1]
        return runs, outs
    if o == S2:
        runs = b[2] + b[1]
        third = 0
        if b[0]:
            if rng.random() < 0.44:
                runs += 1
            else:
                third = 1
        b[:] = [0, 1, third]
        return runs, outs
    if o == S1:
        runs = b[2]
        new = [1, 0, 0]
        if b[1]:
            if rng.random() < 0.60:
                runs += 1
            else:
                new[2] = 1
        if b[0]:
            if not new[2] and rng.random() < 0.28:
                new[2] = 1
            else:
                new[1] = 1
        b[:] = new
        return runs, outs
    # OUT
    if outs < 2 and b[0] and rng.random() < 0.12:       # double play
        b[0] = 0
        return 0, outs + 2
    outs += 1
    if outs < 3:
        if b[2] and rng.random() < 0.30:                 # sac fly / productive out
            b[2] = 0
            runs += 1
        if b[1] and not b[2] and rng.random() < 0.25:
            b[1], b[2] = 0, 1
    return runs, outs


class _Staff:
    def __init__(self, ctx: SimContext, team: str, plan: dict, day: date, fatigue: Fatigue, rng):
        self.t = ctx.teams[team]
        self.rng, self.day, self.fatigue = rng, day, fatigue
        self.pitcher = int(plan["starter"])
        bfs = self.t.starter_bf.get(self.pitcher)
        self.max_bf = int(rng.choice(bfs)) if bfs is not None and len(bfs) else 18
        if plan.get("opener_max_bf"):
            self.max_bf = min(self.max_bf, int(plan["opener_max_bf"]))
        self.bulk = int(plan["bulk"]) if plan.get("bulk") else None
        self.is_starter = True
        self.bf = 0
        self.runs = 0
        self.outs_target = None
        self.outs_got = 0
        self.used = {self.pitcher}
        self.pitch_counts: dict[int, int] = {}

    def record(self, runs: int, outs_added: int):
        self.bf += 1
        self.runs += runs
        self.outs_got += outs_added
        self.pitch_counts[self.pitcher] = self.pitch_counts.get(self.pitcher, 0) + 4

    def maybe_change(self, inning: int, margin: int, inning_start: bool):
        if self.is_starter:
            pull = self.bf >= self.max_bf or self.runs >= 5 or (self.runs >= 4 and self.bf >= 12)
            if not pull:
                return
            if self.bulk and self.bulk not in self.used:
                self._enter(self.bulk, starter_like=True)
                return
            self._reliever(inning, margin)
            return
        done = self.outs_target is not None and self.outs_got >= self.outs_target
        if done or self.runs >= 3:
            self._reliever(inning, margin)

    def _enter(self, pid: int, starter_like: bool = False):
        self.pitcher = pid
        self.used.add(pid)
        self.bf = self.runs = self.outs_got = 0
        if starter_like:
            bfs = self.t.starter_bf.get(pid)
            self.max_bf = int(self.rng.choice(bfs)) if bfs is not None and len(bfs) else 12
            self.is_starter = True
            self.bulk = None
        else:
            self.is_starter = False
            self.outs_target = self.t.usage.outing_outs(pid, self.rng)

    def _reliever(self, inning: int, margin: int):
        avail = [p for p in self.t.bullpen if p not in self.used and self.fatigue.available(p, self.day)]
        if not avail:
            avail = [p for p in self.t.bullpen if p not in self.used] or list(self.t.bullpen)
        pid = self.t.usage.choose(inning, margin, avail, self.rng)
        if pid is None:
            return
        self._enter(pid)

    def close_out(self):
        for p, n in self.pitch_counts.items():
            if p in self.t.bullpen:
                self.fatigue.add(p, self.day, n)


def sim_game(ctx: SimContext, plan: GamePlan, fatigue: Fatigue, rng: np.random.Generator,
             max_innings: int = 20) -> tuple[int, int]:
    """Returns (away_runs, home_runs)."""
    home = plan.home
    away = next(t for t in ctx.teams if t != home)
    staffs = {t: _Staff(ctx, t, plan.starters[t], plan.day, fatigue, rng) for t in (away, home)}
    orders = {}
    for t in (away, home):
        opp = home if t == away else away
        hand = ctx.teams[opp].throws.get(int(plan.starters[opp]["starter"]), "R")
        orders[t] = ctx.teams[t].lineups["vs_" + hand]
    idx = {away: 0, home: 0}
    score = {away: 0, home: 0}
    inning = 1
    while True:
        for half, bat in ((0, away), (1, home)):
            if half == 1 and inning >= 9 and score[home] > score[away]:
                for s in staffs.values():
                    s.close_out()
                return score[away], score[home]
            fld = home if bat == away else away
            st = staffs[fld]
            outs, bases = 0, [0, 0, 0]
            first = True
            while outs < 3:
                st.maybe_change(inning, score[fld] - score[bat], first)
                first = False
                batter = orders[bat][idx[bat] % 9]
                idx[bat] += 1
                c = ctx.cum_probs(batter, st.pitcher, bat, plan.venue, bat == home)
                o = int(np.searchsorted(c, rng.random() * c[-1], side="right"))
                o = min(o, 6)
                before = outs
                runs, outs = _advance(o, bases, outs, rng)
                if outs >= 3:
                    runs = 0 if o == OUT else runs
                score[bat] += runs
                st.record(runs, min(outs, 3) - before)
                if half == 1 and inning >= 9 and score[home] > score[away]:
                    for s in staffs.values():
                        s.close_out()
                    return score[away], score[home]
        if inning >= 9 and score[home] != score[away]:
            for s in staffs.values():
                s.close_out()
            return score[away], score[home]
        inning += 1
        if inning > max_innings:
            for s in staffs.values():
                s.close_out()
            if rng.random() < 0.5:
                score[home] += 1
            else:
                score[away] += 1
            return score[away], score[home]


def sim_series(ctx: SimContext, plans: list[GamePlan], n: int, seed: int,
               wins_needed: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    teams = list(ctx.teams)
    a, b = teams
    lengths: dict[str, int] = {}
    game_wins = np.zeros((len(plans), 2))
    game_played = np.zeros(len(plans))
    runs = {a: [], b: []}
    winners = []
    for _ in range(n):
        fatigue = Fatigue()
        w = {a: 0, b: 0}
        for gi, plan in enumerate(plans):
            ar, hr = sim_game(ctx, plan, fatigue, rng)
            away = a if plan.home == b else b
            win = plan.home if hr > ar else away
            w[win] += 1
            game_played[gi] += 1
            game_wins[gi, teams.index(win)] += 1
            runs[away].append(ar)
            runs[plan.home].append(hr)
            if w[win] == wins_needed:
                key = f"{win} in {gi + 1}"
                lengths[key] = lengths.get(key, 0) + 1
                winners.append(win)
                break
    p = {t: winners.count(t) / n for t in teams}
    return {
        "n": n, "p_win": p,
        "lengths": {k: v / n for k, v in sorted(lengths.items())},
        "game_win_prob": [{"game": i + 1, "played_pct": game_played[i] / n,
                           **{t: (game_wins[i, j] / game_played[i] if game_played[i] else None) for j, t in enumerate(teams)}}
                          for i in range(len(plans))],
        "runs_per_game": {t: float(np.mean(v)) for t, v in runs.items()},
    }


def calibrate(ctx: SimContext, plans: list[GamePlan], n_games: int = 1500, seed: int = 0) -> dict:
    """Check the simulator's scoring level against reality.

    Runs three variants on the real series schedule (parks, home tilt and bullpen usage included):
      league : every batter and pitcher at league-average rates
      offense: each team's real hitters vs league-average pitching
      defense: each team's real pitchers vs league-average hitting
    Compare the results with actual runs per game to see whether the simulator scores too much or too
    little, and whether each team's offense and run prevention come through at the right size.
    """
    from .pa_model import Rates  # local import to avoid a cycle at module load

    orig_rates, orig_cache = ctx.rates, ctx.cache
    r = orig_rates
    variants = {
        "league": Rates(r.league, {}, {}, {}, {}),
        "offense": Rates(r.league, r.batter, {}, r.batter_n, {}),
        "defense": Rates(r.league, {}, r.pitcher, {}, r.pitcher_n),
    }
    out = {}
    teams = list(ctx.teams)
    for name, rates in variants.items():
        ctx.rates, ctx.cache = rates, {}
        rng = np.random.default_rng(seed)
        runs = {t: [] for t in teams}
        for i in range(n_games):
            plan = plans[i % len(plans)]
            ar, hr = sim_game(ctx, plan, Fatigue(), rng)
            away = next(t for t in teams if t != plan.home)
            runs[away].append(ar)
            runs[plan.home].append(hr)
        out[name] = {t: float(np.mean(v)) for t, v in runs.items()}
    ctx.rates, ctx.cache = orig_rates, orig_cache
    return out


def calibrate_scoring(ctx: SimContext, plans: list[GamePlan], target_rpg: float, n_games: int = 800,
                      seed: int = 0) -> dict:
    """Set ctx.scoring_tilt so league-average teams score target_rpg in the simulator.

    The simulator leaves out stolen bases, wild pitches, errors and other small run sources, so it
    scores a little low. A single tilt on the "good for the hitter" outcomes closes the gap without
    changing who is better than whom. Two probe runs, then log-linear interpolation.
    """
    from .pa_model import Rates

    orig_rates, orig_cache, orig_tilt = ctx.rates, ctx.cache, ctx.scoring_tilt
    ctx.rates = Rates(orig_rates.league, {}, {}, {}, {})

    def rpg(tilt: float) -> float:
        ctx.scoring_tilt, ctx.cache = tilt, {}
        rng = np.random.default_rng(seed)
        tot = 0
        for i in range(n_games):
            ar, hr = sim_game(ctx, plans[i % len(plans)], Fatigue(), rng)
            tot += ar + hr
        return tot / (2 * n_games)

    t0, t1 = 0.0, 0.10
    r0, r1 = rpg(t0), rpg(t1)
    slope = (np.log(r1) - np.log(r0)) / (t1 - t0)
    tilt = float(t0 + (np.log(target_rpg) - np.log(r0)) / slope) if slope > 0 else 0.0
    tilt = float(np.clip(tilt, -0.3, 0.3))
    check = rpg(tilt)
    ctx.rates, ctx.cache, ctx.scoring_tilt = orig_rates, {}, tilt
    return {"target_rpg": target_rpg, "rpg_at_0": r0, "tilt": tilt, "rpg_after": check}
