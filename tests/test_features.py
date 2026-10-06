import numpy as np

from alcs_model.features import OUTCOMES, leverage_table


def test_prepare_adds_columns(pitches):
    for c in ("bat_team", "fld_team", "pgroup", "is_swing", "is_whiff", "outcome", "base_state", "xrv"):
        assert c in pitches.columns
    assert set(pitches["bat_team"].unique()) <= {"CWS", "TB", "CLE", "NYY", "HOU", "BOS"}


def test_one_row_per_pa(pa):
    assert not pa.duplicated(["game_pk", "at_bat_number"]).any()
    assert pa["outcome"].isin(OUTCOMES).all()


def test_xrv_is_centered(pitches):
    bip = pitches["is_bip"]
    assert abs(pitches.loc[bip, "xrv"].mean() - pitches.loc[bip, "delta_run_exp"].mean()) < 0.05


def test_leverage_averages_near_one(pa):
    t = leverage_table(pa)
    w = np.average(t["li"], weights=t["n_f"])
    assert 0.8 < w < 1.2
