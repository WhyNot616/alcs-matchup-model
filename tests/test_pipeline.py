import json

import pytest

import alcs_model.config as C
import alcs_model.data as D
import alcs_model.pipeline as P


@pytest.fixture
def tmp_project(tmp_path, monkeypatch, raw):
    raw_dir, out, docs = tmp_path / "data" / "raw", tmp_path / "output", tmp_path / "docs"
    raw_dir.mkdir(parents=True)
    for mod in (C, D, P):
        for name, val in (("DATA_RAW", raw_dir), ("OUTPUT", out), ("DOCS", docs)):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, val)
    monkeypatch.setattr(C, "DATA_PROCESSED", tmp_path / "data" / "processed")
    raw.to_parquet(raw_dir / "statcast_league_2026.parquet", index=False)
    return tmp_path


def test_end_to_end(tmp_project, cfg):
    data = P.run(cfg, n_series=40, n_boot=2, seed=1)
    for key in ("team", "hitters", "pitchers", "records", "matrix", "bullpen", "sim", "backtest"):
        assert key in data
    p = data["sim"]["p_win"]
    assert abs(sum(p.values()) - 1) < 1e-9
    assert data["sim"]["bootstrap"]["n"] == 2
    html = (tmp_project / "docs" / "index.html").read_text()
    assert "__DATA__" not in html and len(html) > 10_000
    saved = json.loads((tmp_project / "output" / "dashboard_data.json").read_text())
    assert saved["meta"]["season"] == 2026
