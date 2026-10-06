"""Game-level backtest: predict real games using only what was known before first pitch.

Two test sets:
  postseason : every final 2026 playoff game so far (Wild Card, Division Series, ...), predicted from a
               model fit on the full regular season.
  regular    : every regular-season game on or after model.backtest_split, predicted from a model fit
               only on games before the split. Hundreds of games, so it has real statistical power.

For each game the simulator gets that game's actual batting orders and starting pitchers (known
pregame), each team's bullpen as it stood in the 30 days before the game, each manager's relief usage
pattern from the training data, and bullpen fatigue from the real pitch counts of the previous two days.
Nothing from the game itself or any later game is used.

Baselines, all pregame and leak-free:
  coin flip, home field only, log5 of team win %, and log5 of Pythagorean win % (both with home field).
Betting-market closing lines are the strongest benchmark but are not pulled here.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .bullpen import UsageModel, appearances
from .config import OUTPUT, Config, ensure_dirs
from .data import TEAM_IDS, load_leaderboard, load_postseason_games, pull_postseason_games
from .features import OUTCOMES, leverage_table
from .matchups import build_profiles, pair_edges
from .pa_model import Rates, _logloss, _predict, batter_side, build_rates
from .simulate import Fatigue, GamePlan, SimContext, TeamSetup, sim_game

ID2ABBR = {v: k for k, v in TEAM_IDS.items()}
POST_LABEL = {"F": "Wild Card", "D": "Division Series", "L": "LCS", "W": "World Series"}


@dataclass
class GameSpec:
    game_pk: int
    day: date
    game_type: str
    label: str
    home: str
    away: str
    lineup: dict[str, list[int]]
    starter: dict[str, int]
    home_runs: int
    away_runs: int
    venue: str | None = None
    pred: dict = field(default_factory=dict)


# ---------------------------------------------------------------- park tables for every venue
def park_tables(pf: pd.DataFrame | None):
    """(venue, side) -> outcome multipliers, team -> home venue, venue -> runs index."""
    mult, venue_of, runs = {}, {}, {}
    if pf is None or pf.empty:
        return mult, venue_of, runs
    for years in (1, 3):  # 3-year values overwrite 1-year where both exist
        sub = pf[pf["rolling_years"] == years]
        for _, r in sub.iterrows():
            v, side = r["venue_name"], str(r["bat_side_key"])
            if side == "All":
                try:
                    venue_of[ID2ABBR[int(float(r["main_team_id"]))]] = v
                except (KeyError, ValueError, TypeError):
                    pass
                if pd.notna(r.get("index_runs")):
                    runs[v] = float(r["index_runs"]) / 100
            else:
                def f(c, r=r):
                    return float(r[c]) / 100 if pd.notna(r[c]) else 1.0
                mult[(v, side)] = np.array([f("index_so"), f("index_bb"), f("index_1b"), f("index_2b"),
                                            f("index_3b"), f("index_hr"), 1.0])
    return mult, venue_of, runs


# ---------------------------------------------------------------- game tables
def game_scores(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("game_pk").agg(date=("game_date", "first"), game_type=("game_type", "first"),
                                 home=("home_team", "first"), away=("away_team", "first"),
                                 hr=("post_home_score", "max"), ar=("post_away_score", "max"))
    g["game_type"] = g["game_type"].astype(str)
    return g


def specs_from_statcast(p: pd.DataFrame, game_pks) -> list[GameSpec]:
    d = p[p["game_pk"].isin(set(game_pks))].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    first_bat = d.drop_duplicates(["game_pk", "bat_team", "batter"])
    lineups = first_bat.groupby(["game_pk", "bat_team"])["batter"].apply(lambda s: [int(x) for x in s.head(9)])
    starters = d.drop_duplicates(["game_pk", "fld_team"]).set_index(["game_pk", "fld_team"])["pitcher"]
    sc = game_scores(d)
    out = []
    for pk, r in sc.iterrows():
        h, a = str(r["home"]), str(r["away"])
        try:
            lu = {h: lineups.loc[(pk, h)], a: lineups.loc[(pk, a)]}
            sp = {h: int(starters.loc[(pk, h)]), a: int(starters.loc[(pk, a)])}
        except KeyError:
            continue
        if len(lu[h]) < 9 or len(lu[a]) < 9:
            continue
        gt = str(r["game_type"])
        out.append(GameSpec(int(pk), pd.Timestamp(r["date"]).date(), gt, POST_LABEL.get(gt, "Regular season"),
                            h, a, lu, sp, int(r["hr"]), int(r["ar"])))
    return out


def specs_from_boxscores(rows: list[dict]) -> list[GameSpec]:
    out = []
    for g in rows:
        h, a = g["home"]["team"], g["away"]["team"]
        if not h or not a or len(g["home"]["lineup"]) < 9 or len(g["away"]["lineup"]) < 9:
            continue
        if not g["home"]["pitchers"] or not g["away"]["pitchers"] or g.get("home_runs") is None:
            continue
        out.append(GameSpec(int(g["game_pk"]), date.fromisoformat(g["date"]), g["game_type"],
                            g.get("label") or POST_LABEL.get(g["game_type"], ""), h, a,
                            {h: g["home"]["lineup"], a: g["away"]["lineup"]},
                            {h: int(g["home"]["pitchers"][0]), a: int(g["away"]["pitchers"][0])},
                            int(g["home_runs"]), int(g["away_runs"]), g.get("venue")))
    return out


# ---------------------------------------------------------------- the trained "world"
class World:
    """Everything the simulator needs, fit on a training window."""

    def __init__(self, cfg: Config, p_all: pd.DataFrame, p_train: pd.DataFrame, pa_train: pd.DataFrame,
                 tuned: dict | None, lam: float, pf: pd.DataFrame | None, verbose: bool = True):
        m = cfg.model
        self.as_of = p_train["game_date"].max()
        hl = m.get("recent_half_life_days", 0)
        k = dict(m["shrink"]["pa"])
        if tuned:
            hl = tuned.get("half_life", hl)
            k = {o: v * tuned.get("shrink_mult", 1.0) for o, v in k.items()}
        self.rates = build_rates(pa_train, k, hl, self.as_of)
        self.lam = lam
        self.profiles = build_profiles(p_train, m["shrink"], hl, self.as_of) if lam else None
        self.li = leverage_table(pa_train)
        self.park, self.venue_of, self.runs_idx = park_tables(pf)
        self.home_tilt = float(m.get("home_pa_tilt", 0.05))
        self.scoring_tilt = 0.0
        self.cache: dict = {}

        # bats / throws from all data (static player facts)
        st = p_all.dropna(subset=["batter", "stand"]).groupby("batter")["stand"].agg(
            lambda s: "S" if s.nunique() > 1 else s.iloc[0])
        self.bats = {int(k_): str(v) for k_, v in st.items()}
        th = p_all.dropna(subset=["pitcher", "p_throws"]).groupby("pitcher")["p_throws"].first()
        self.throws = {int(k_): str(v) for k_, v in th.items()}

        # usage models and outing lengths from training data
        self.usage: dict[str, UsageModel] = {}
        apps = []
        for t in sorted(set(p_train["fld_team"].dropna().astype(str))):
            app = appearances(p_train, t, self.li)
            rel = [int(x) for x in app.loc[~app["is_start"], "pitcher"].unique()]
            self.usage[t] = UsageModel(app, rel, boost=1.0)
            apps.append(app)
        A = pd.concat(apps, ignore_index=True)
        self.starter_bf = {}
        for pid, g in A.groupby("pitcher"):
            st_bf = g.loc[g["is_start"], "bf"].to_numpy()
            self.starter_bf[int(pid)] = st_bf if len(st_bf) >= 3 else g["bf"].to_numpy()

        # pregame roster and fatigue facts from all data (only dates before each game are used)
        d = p_all.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        a = d.groupby(["game_pk", "fld_team", "pitcher"], sort=False).agg(
            game_date=("game_date", "first"), first_ab=("at_bat_number", "min"), pitches=("pitch_number", "size"))
        a = a.reset_index()
        a["is_start"] = a["first_ab"].eq(a.groupby(["game_pk", "fld_team"])["first_ab"].transform("min"))
        a["day"] = a["game_date"].dt.date
        self.app_light = a

        hg = game_scores(p_train)
        hg = hg[hg["game_type"].eq("R")]
        self.league_rpg = float((hg["hr"].mean() + hg["ar"].mean()) / 2)
        self.home_win = float((hg["hr"] > hg["ar"]).mean())
        rec = {}
        for t in set(hg["home"]) | set(hg["away"]):
            hm, aw = hg[hg["home"].eq(t)], hg[hg["away"].eq(t)]
            w = int((hm["hr"] > hm["ar"]).sum() + (aw["ar"] > aw["hr"]).sum())
            gp = len(hm) + len(aw)
            rf = float(hm["hr"].sum() + aw["ar"].sum())
            ra = float(hm["ar"].sum() + aw["hr"].sum())
            rec[str(t)] = {"w": w, "g": gp, "rf": rf, "ra": ra}
        self.records = rec
        if verbose:
            print(f"  world fit through {self.as_of.date()}: {len(pa_train):,} PA, "
                  f"league {self.league_rpg:.2f} R/G, home win {self.home_win:.3f}", flush=True)

    # -- pregame facts for one game
    def bullpen(self, team: str, day: date, exclude: int) -> list[int]:
        a = self.app_light
        w = a[a["fld_team"].eq(team) & (a["day"] < day) & (a["day"] >= day - timedelta(days=30)) & ~a["is_start"]]
        return [int(x) for x in w["pitcher"].unique() if int(x) != exclude]

    def fatigue(self, teams: list[str], day: date) -> list[tuple[int, date, int]]:
        a = self.app_light
        w = a[a["fld_team"].isin(teams) & (a["day"] < day) & (a["day"] >= day - timedelta(days=2))]
        return [(int(r.pitcher), r.day, int(r.pitches)) for r in w.itertuples()]

    def venue(self, spec: GameSpec) -> str:
        if spec.venue and any(k[0] == spec.venue for k in self.park):
            return spec.venue
        return self.venue_of.get(spec.home, spec.venue or spec.home)

    def context(self, spec: GameSpec, boost: float, rates: Rates | None = None, cache: dict | None = None):
        teams = {}
        for t in (spec.home, spec.away):
            usage = self.usage.get(t) or next(iter(self.usage.values()))  # every MLB team is in training data
            usage.boost = boost
            teams[t] = TeamSetup(t, {"vs_R": spec.lineup[t], "vs_L": spec.lineup[t]}, self.bats, self.throws,
                                 self.bullpen(t, spec.day, spec.starter[t]), usage, self.starter_bf,
                                 self.venue_of.get(t, ""))
        edges = {}
        if self.lam and self.profiles is not None:
            B, P, S = [], [], []
            for t in (spec.home, spec.away):
                o = spec.away if t == spec.home else spec.home
                arms = [spec.starter[o]] + teams[o].bullpen
                for b in spec.lineup[t]:
                    for p in arms:
                        B.append(b)
                        P.append(p)
                        S.append(batter_side(self.bats.get(b, "R"), self.throws.get(p, "R")))
            e = pair_edges(self.profiles, np.array(B), np.array(P), np.array(S))
            edges = {(b, p): float(x) for b, p, x in zip(B, P, e)}
        ctx = SimContext(rates or self.rates, edges, self.lam, self.park, self.home_tilt, teams,
                         cache=self.cache if cache is None else cache, scoring_tilt=self.scoring_tilt)
        plan = GamePlan(spec.home, self.venue(spec), {spec.home: {"starter": spec.starter[spec.home]},
                                                      spec.away: {"starter": spec.starter[spec.away]}}, spec.day)
        return ctx, plan

    def simulate(self, spec: GameSpec, n: int, seed: int, boost: float = 1.0, rates: Rates | None = None,
                 cache: dict | None = None) -> dict:
        ctx, plan = self.context(spec, boost, rates, cache)
        seeds = self.fatigue([spec.home, spec.away], spec.day)
        rng = np.random.default_rng(seed)
        hw, hr_sum, ar_sum, tie = 0, 0, 0, 0
        for _ in range(n):
            fat = Fatigue()
            for pid, d_, k in seeds:
                fat.add(pid, d_, k)
            ar, hr = sim_game(ctx, plan, fat, rng)
            hw += hr > ar
            hr_sum += hr
            ar_sum += ar
        return {"p_home": hw / n, "exp_home": hr_sum / n, "exp_away": ar_sum / n}

    def calibrate_scoring(self, specs: list[GameSpec], n_per: int = 12, seed: int = 0, verbose: bool = True) -> dict:
        """Pick the global scoring tilt so league-average players score like the training league did."""
        lg_rates = Rates(self.rates.league, {}, {}, {}, {})
        sample = specs if len(specs) <= 150 else [specs[i] for i in np.linspace(0, len(specs) - 1, 150).astype(int)]
        target = self.league_rpg * float(np.mean([self.runs_idx.get(self.venue(s), 1.0) for s in sample]))

        def rpg(tilt):
            self.scoring_tilt = tilt
            cache: dict = {}
            tot = 0.0
            for i, s in enumerate(sample):
                r = self.simulate(s, n_per, seed + i, rates=lg_rates, cache=cache)
                tot += r["exp_home"] + r["exp_away"]
            return tot / (2 * len(sample))
        r0, r1 = rpg(0.0), rpg(0.1)
        slope = (np.log(r1) - np.log(r0)) / 0.1
        tilt = float(np.clip((np.log(target) - np.log(r0)) / slope, -0.3, 0.3)) if slope > 0 else 0.0
        self.scoring_tilt = tilt
        self.cache = {}
        if verbose:
            print(f"  scoring tilt {tilt:+.3f} (league-average sim {r0:.2f} R/G, target {target:.2f})", flush=True)
        return {"tilt": tilt, "rpg_at_0": r0, "target": target}

    # -- baselines
    def baselines(self, spec: GameSpec) -> dict:
        hw = self.home_win
        ho = hw / (1 - hw)

        def log5(pa, pb):
            pa, pb = np.clip(pa, .2, .8), np.clip(pb, .2, .8)
            o = (pa / (1 - pa)) / (pb / (1 - pb)) * ho
            return float(o / (1 + o))
        rh, ra = self.records.get(spec.home), self.records.get(spec.away)
        out = {"coin": 0.5, "home": hw}
        if rh and ra and rh["g"] and ra["g"]:
            out["log5_wpct"] = log5(rh["w"] / rh["g"], ra["w"] / ra["g"])
            py = lambda r: r["rf"] ** 1.83 / (r["rf"] ** 1.83 + r["ra"] ** 1.83)  # noqa: E731
            out["log5_pythag"] = log5(py(rh), py(ra))
        return out


# ---------------------------------------------------------------- scoring
def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    acc = np.where(p == 0.5, 0.5, ((p > 0.5) == (y == 1)).astype(float))
    return {"n": int(len(y)), "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "accuracy": float(np.mean(acc))}


def score_games(games: list[GameSpec], seed: int = 0, n_boot: int = 2000) -> dict:
    if not games:
        return {}
    y = np.array([1.0 if g.home_runs > g.away_runs else 0.0 for g in games])
    names = ["model"] + [k for k in games[0].pred["baselines"]]
    preds = {"model": np.array([g.pred["p_home"] for g in games])}
    for k in names[1:]:
        preds[k] = np.array([g.pred["baselines"].get(k, np.nan) for g in games])
    out = {"metrics": {}, "vs_model": {}}
    for k, v in preds.items():
        ok = ~np.isnan(v)
        out["metrics"][k] = _metrics(v[ok], y[ok])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    bm = (preds["model"] - y) ** 2
    for k in names[1:]:
        v = preds[k]
        if np.isnan(v).any():
            continue
        diff = bm - (v - y) ** 2                     # negative = model better
        boots = diff[idx].mean(1)
        out["vs_model"][k] = {"brier_diff": float(diff.mean()), "lo": float(np.percentile(boots, 2.5)),
                              "hi": float(np.percentile(boots, 97.5)),
                              "p_model_better": float((boots < 0).mean())}
    exp_tot = np.array([g.pred["exp_home"] + g.pred["exp_away"] for g in games])
    act_tot = np.array([g.home_runs + g.away_runs for g in games], dtype=float)
    exp_m = np.array([g.pred["exp_home"] - g.pred["exp_away"] for g in games])
    act_m = np.array([g.home_runs - g.away_runs for g in games], dtype=float)
    out["runs"] = {"exp_total_mean": float(exp_tot.mean()), "act_total_mean": float(act_tot.mean()),
                   "rmse_total": float(np.sqrt(np.mean((exp_tot - act_tot) ** 2))),
                   "rmse_total_const": float(np.sqrt(np.mean((exp_tot.mean() - act_tot) ** 2))),
                   "corr_margin": float(np.corrcoef(exp_m, act_m)[0, 1]) if len(games) > 2 and np.std(exp_m) > 0 else None}
    if len(games) >= 100:
        bins = [0, .35, .40, .45, .50, .55, .60, .65, 1.0]
        b = pd.cut(preds["model"], bins, include_lowest=True)
        cal = pd.DataFrame({"p": preds["model"], "y": y, "b": b}).groupby("b", observed=True).agg(
            pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
        out["calibration"] = cal.reset_index(drop=True).round(4).to_dict(orient="records")
        fav = np.where(preds["model"] >= 0.5, preds["model"], 1 - preds["model"])
        fav_won = np.where(preds["model"] >= 0.5, y, 1 - y)
        out["favorites"] = {"mean_fav_prob": float(fav.mean()), "fav_win_rate": float(fav_won.mean())}
    return out


def pa_check(world: World, pa_test: pd.DataFrame) -> dict:
    """Plate-appearance log loss on the test PAs: talent model vs league rates."""
    t = pa_test[pa_test["outcome"].isin(OUTCOMES)].copy()
    if len(t) < 50:
        return {}
    t["stand"] = t["stand"].astype(str)
    t["p_throws"] = t["p_throws"].astype(str)
    y = t["outcome"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    P = _predict(world.rates, t, None, 0.0)
    lg = np.vstack([world.rates.league.get((s, h), np.mean(list(world.rates.league.values()), axis=0))
                    for s, h in zip(t["stand"], t["p_throws"])])
    a, b = _logloss(lg, y), _logloss(P, y)
    return {"n": int(len(t)), "logloss_league": a, "logloss_model": b, "skill_pct": 100 * (a - b) / a}


# ---------------------------------------------------------------- driver
def game_backtest(cfg: Config, p: pd.DataFrame, pa: pd.DataFrame, pf: pd.DataFrame | None = None,
                  post_rows: list[dict] | None = None, post_sims: int = 4000, reg_sims: int = 300,
                  include_regular: bool = True, seed: int = 7, verbose: bool = True) -> dict:
    bt_path = OUTPUT / "backtest.json"
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else None
    tuned = bt.get("tuned") if bt else None
    lam = float(bt["lambda"]) if bt and bt.get("mix_gain_pct", 0) > 0 else 0.0
    boost = float(cfg.model.get("postseason_leverage_boost", 1.0))
    t0 = time.time()
    gt = p["game_type"].astype(str)
    pa_gt = pa["game_type"].astype(str)
    result = {"generated": time.strftime("%Y-%m-%d %H:%M"), "settings": {"post_sims": post_sims, "reg_sims": reg_sims,
              "tuned": tuned, "lambda": lam, "postseason_boost": boost}}

    # ---- postseason
    reg_p, reg_pa = p[gt.eq("R")], pa[pa_gt.eq("R")]
    W = World(cfg, p, reg_p, reg_pa, tuned, lam, pf, verbose)
    post_pks = set(p.loc[~gt.eq("R"), "game_pk"])
    specs = {s.game_pk: s for s in specs_from_statcast(p, post_pks)}
    for s in specs_from_boxscores(post_rows or []):    # boxscores fill gaps and are the source of truth for scores
        if s.game_pk in specs:
            specs[s.game_pk].home_runs, specs[s.game_pk].away_runs = s.home_runs, s.away_runs
            specs[s.game_pk].label, specs[s.game_pk].venue = s.label or specs[s.game_pk].label, s.venue
        else:
            specs[s.game_pk] = s
    post = sorted(specs.values(), key=lambda s: (s.day, s.game_pk))
    if post:
        result["postseason_calibration"] = W.calibrate_scoring(post, seed=seed, verbose=verbose)
        for i, s in enumerate(post):
            s.pred = W.simulate(s, post_sims, seed + i, boost=boost)
            s.pred["baselines"] = W.baselines(s)
            if verbose:
                print(f"  {s.day} {s.label:<28} {s.away}@{s.home}: P(home) {s.pred['p_home']:.3f}  "
                      f"final {s.away_runs}-{s.home_runs}", flush=True)
        result["postseason"] = {"games": [_game_row(s, W) for s in post], **score_games(post, seed)}
        result["postseason"]["pa"] = pa_check(W, pa[~pa_gt.eq("R")])
        _note(f"postseason: {len(post)} games, model Brier "
              f"{result['postseason']['metrics']['model']['brier']:.4f}", t0)
    else:
        result["postseason"] = {"games": []}

    # ---- late regular season
    if include_regular:
        split = pd.Timestamp(cfg.model["backtest_split"])
        tr_p, tr_pa = p[gt.eq("R") & (p["game_date"] < split)], pa[pa_gt.eq("R") & (pa["game_date"] < split)]
        test_pks = set(p.loc[gt.eq("R") & (p["game_date"] >= split), "game_pk"])
        W2 = World(cfg, p, tr_p, tr_pa, tuned, lam, pf, verbose)
        reg = sorted(specs_from_statcast(p, test_pks), key=lambda s: (s.day, s.game_pk))
        result["regular_calibration"] = W2.calibrate_scoring(reg, seed=seed, verbose=verbose)
        for i, s in enumerate(reg):
            s.pred = W2.simulate(s, reg_sims, seed + 1000 + i, boost=1.0)
            s.pred["baselines"] = W2.baselines(s)
            if verbose and (i % 100 == 0 or i == len(reg) - 1):
                _note(f"regular season: {i + 1}/{len(reg)} games simulated", t0)
        sc = score_games(reg, seed)
        sc["pa"] = pa_check(W2, pa[pa_gt.eq("R") & (pa["game_date"] >= split)])
        result["regular"] = {"split": str(split.date()), "n_games": len(reg), **sc}
        _note(f"regular season: {len(reg)} games, model Brier {sc['metrics']['model']['brier']:.4f}, "
              f"best baseline " + ", ".join(f"{k} {v['brier']:.4f}" for k, v in sc["metrics"].items() if k != "model"), t0)
    return result


def _game_row(s: GameSpec, W: World) -> dict:
    return {"game_pk": s.game_pk, "date": str(s.day), "label": s.label, "home": s.home, "away": s.away,
            "starters": {s.home: s.starter[s.home], s.away: s.starter[s.away]},
            "p_home": round(s.pred["p_home"], 4), "exp_home": round(s.pred["exp_home"], 2),
            "exp_away": round(s.pred["exp_away"], 2), "home_runs": s.home_runs, "away_runs": s.away_runs,
            "baselines": {k: round(v, 4) for k, v in s.pred["baselines"].items()}, "venue": W.venue(s)}


def _note(msg: str, t0: float) -> None:
    line = f"[{time.time() - t0:6.1f}s] {msg}"
    print(line, flush=True)
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::notice title=game backtest::{line}", flush=True)


def run(cfg: Config, post_sims: int = 4000, reg_sims: int = 300, include_regular: bool = True,
        seed: int = 7) -> dict:
    from .data import lookup_names
    from .pipeline import load_prepared

    ensure_dirs()
    p, pa = load_prepared(cfg)
    try:
        pull_postseason_games(cfg)
    except Exception as e:  # boxscores are a supplement; Statcast alone still works
        print(f"  warning: postseason boxscores failed ({e})")
    res = game_backtest(cfg, p, pa, load_leaderboard(cfg, "park_factors"), load_postseason_games(cfg),
                        post_sims, reg_sims, include_regular, seed)
    ids = {pid for g in res["postseason"]["games"] for pid in g["starters"].values()}
    names = lookup_names(ids) if ids else {}
    for g in res["postseason"]["games"]:
        g["starter_names"] = {t: names.get(pid, str(pid)) for t, pid in g["starters"].items()}
    (OUTPUT / "game_backtest.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUTPUT / 'game_backtest.json'}")
    return res
