import numpy as np

from alcs_model.matchups import build_profiles, matchup_detail, pair_edges
from alcs_model.pa_model import backtest, build_rates, expected_woba, matchup_probs


def test_probs_sum_to_one(cfg, pa):
    r = build_rates(pa, cfg.model["shrink"]["pa"])
    b, p = int(pa["batter"].iloc[0]), int(pa["pitcher"].iloc[0])
    pr = matchup_probs(r, b, "R", p, "R", edge=1.0, lam=0.05)
    assert abs(pr.sum() - 1) < 1e-9 and (pr > 0).all()


def test_positive_edge_raises_xwoba(cfg, pa):
    r = build_rates(pa, cfg.model["shrink"]["pa"])
    b, p = int(pa["batter"].iloc[0]), int(pa["pitcher"].iloc[0])
    lo = expected_woba(matchup_probs(r, b, "R", p, "R", edge=-1.0, lam=0.05))
    hi = expected_woba(matchup_probs(r, b, "R", p, "R", edge=1.0, lam=0.05))
    assert hi > lo


def test_profiles_and_edges(cfg, pitches):
    prof = build_profiles(pitches, cfg.model["shrink"])
    assert not prof.hitter.empty and not prof.pitcher.empty
    u = prof.usage.groupby(level=[0, 1])["u"].sum()
    assert np.allclose(u.to_numpy(), 1.0)
    d = pitches.drop_duplicates(["batter", "pitcher"]).head(50)
    e = pair_edges(prof, d["batter"].to_numpy(), d["pitcher"].to_numpy(), d["stand"].astype(str).to_numpy())
    assert len(e) == 50 and np.isfinite(e).all()
    det = matchup_detail(prof, int(d["batter"].iloc[0]), int(d["pitcher"].iloc[0]), str(d["stand"].iloc[0]))
    assert det and abs(sum(x["u"] for x in det) - 1) < 1e-3


def test_backtest_runs(cfg, pitches, pa):
    res = backtest(pitches, pa, cfg.model, verbose=False)
    ll = res["logloss"]
    assert set(ll) == {"league", "oddsratio", "oddsratio_mix", "with_stuff"}
    assert "gamma" in res["stuff"] and "stability" in res["stuff"]
    # synthetic hitters/pitchers have real talent differences, so talent should beat league-only
    assert ll["oddsratio"] < ll["league"]
