import numpy as np

from alcs_model.stuff import fit_stuff, stability_check


def test_stuff_model_rates_pitchers(pitches):
    p = pitches.copy()
    # plant a physical signal: faster fastballs produce better (lower) run values for the pitcher
    rng = np.random.default_rng(0)
    p["pfx_x"] = rng.normal(0, 0.8, len(p))
    p["pfx_z"] = rng.normal(1.0, 0.5, len(p))
    p["release_spin_rate"] = rng.normal(2300, 150, len(p))
    fast = p["release_speed"] > p["release_speed"].median()
    p.loc[fast, "xrv"] -= 0.03
    sm = fit_stuff(p, verbose=False)
    r = sm.ratings
    assert len(r) > 10 and np.isfinite(r["stuff_plus"]).all()
    velo = p.groupby("pitcher")["release_speed"].mean().reindex(r.index)
    assert np.corrcoef(velo, r["rv100"])[0, 1] > 0.3   # harder throwers rate better


def test_stability_check_runs(pitches):
    res = stability_check(pitches, pitches["game_date"].quantile(0.5), min_pitches=100)
    assert res["n"] >= 0
