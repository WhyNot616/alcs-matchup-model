import numpy as np

from alcs_model.simulate import _advance


def test_advance_rules():
    rng = np.random.default_rng(0)
    b = [1, 1, 1]
    runs, outs = _advance(5, b, 0, rng)       # grand slam
    assert runs == 4 and b == [0, 0, 0] and outs == 0
    b = [1, 1, 1]
    runs, outs = _advance(1, b, 1, rng)       # bases-loaded walk
    assert runs == 1 and b == [1, 1, 1]
    b = [0, 0, 0]
    runs, outs = _advance(0, b, 2, rng)       # strikeout
    assert outs == 3 and runs == 0
