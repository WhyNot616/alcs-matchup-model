import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("ALCS_OFFLINE", "1")

from synth import make_league  # noqa: E402

from alcs_model.config import load_config  # noqa: E402
from alcs_model.features import plate_appearances, prepare_pitches  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return make_league(n_days=150, start=__import__("datetime").date(2026, 4, 15))


@pytest.fixture(scope="session")
def pitches(raw):
    return prepare_pitches(raw)


@pytest.fixture(scope="session")
def pa(pitches):
    return plate_appearances(pitches)


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c.raw["model"]["backtest_split"] = "2026-08-01"
    return c
