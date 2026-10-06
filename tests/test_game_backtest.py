import numpy as np

from alcs_model.backtest_games import game_backtest, specs_from_statcast
from alcs_model.features import plate_appearances, prepare_pitches


def _with_postseason(raw):
    r = raw.copy()
    last = sorted(r["game_date"].unique())[-4:]
    r.loc[r["game_date"].isin(last), "game_type"] = "D"
    return r


def test_specs_have_lineups_and_starters(pitches):
    specs = specs_from_statcast(pitches, pitches["game_pk"].unique()[:20])
    assert specs
    for s in specs:
        assert len(s.lineup[s.home]) == 9 and len(s.lineup[s.away]) == 9
        assert s.starter[s.home] != s.starter[s.away]


def test_game_backtest_runs(raw, cfg, tmp_path, monkeypatch):
    import alcs_model.backtest_games as BG
    monkeypatch.setattr(BG, "OUTPUT", tmp_path)
    p = prepare_pitches(_with_postseason(raw))
    pa = plate_appearances(p)
    cfg.raw["model"]["backtest_split"] = "2026-08-20"
    res = game_backtest(cfg, p, pa, post_sims=60, reg_sims=20, verbose=False)
    games = res["postseason"]["games"]
    assert len(games) >= 6
    assert all(0 < g["p_home"] < 1 for g in games)
    m = res["postseason"]["metrics"]
    assert {"model", "coin", "home", "log5_wpct", "log5_pythag"} <= set(m)
    reg = res["regular"]
    assert reg["n_games"] > 50 and 0 < reg["metrics"]["model"]["brier"] < 0.35
    # synthetic teams have real talent gaps, so the model should beat a coin flip on many games
    assert reg["metrics"]["model"]["brier"] < reg["metrics"]["coin"]["brier"]
    assert np.isfinite(reg["runs"]["rmse_total"])
