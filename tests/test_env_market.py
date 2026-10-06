import numpy as np
import pandas as pd

from alcs_model.data import _ml_prob, _parse_pick
from alcs_model.market import logistic
from alcs_model.postseason_env import expected_runs, fit_factor


def test_moneyline_conversion():
    assert abs(_ml_prob("-150") - 0.6) < 1e-9
    assert abs(_ml_prob("+150") - 0.4) < 1e-9
    pick = [{"provider": {"name": "DraftKings"},
             "moneyline": {"home": {"close": {"odds": "-143"}, "open": {"odds": "-133"}},
                           "away": {"close": {"odds": "+119"}, "open": {"odds": "+110"}}},
             "total": {"over": {"close": {"line": "o7", "odds": "-107"}}}}]
    L = _parse_pick(pick)
    assert 0.55 < L["p_home_close"] < 0.57 and L["total_close"] == 7.0
    assert abs(L["p_home_close"] + (1 - L["p_home_close"]) - 1) < 1e-9


def _season(rng, season, post_scale):
    teams = list("ABCDEFGH")
    rows = []
    for i in range(800):
        h, a = rng.choice(teams, 2, replace=False)
        rows.append((season, i, "R", h, a, rng.poisson(4.5), rng.poisson(4.5)))
    for i in range(40):
        h, a = rng.choice(teams[:4], 2, replace=False)
        rows.append((season, 10_000 + i, "D", h, a, rng.poisson(4.5 * post_scale), rng.poisson(4.5 * post_scale)))
    return pd.DataFrame(rows, columns=["season", "game_pk", "game_type", "home", "away", "home_runs", "away_runs"])


def test_factor_recovers_planted_environment():
    rng = np.random.default_rng(0)
    df = pd.concat([_season(rng, y, 0.85) for y in range(2019, 2025)], ignore_index=True)
    res = fit_factor(expected_runs(df))
    assert res["lo"] < 0.85 < res["hi"] and res["games"] == 240


def test_logistic_recovers_coefficients():
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(0.3 + 1.2 * x)))).astype(float)
    b, se = logistic(np.column_stack([np.ones_like(x), x]), y)
    assert abs(b[1] - 1.2) < 4 * se[1] and abs(b[0] - 0.3) < 4 * se[0]
