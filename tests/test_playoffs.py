from datetime import date, timedelta

import pytest

from alcs_model.features import plate_appearances, prepare_pitches
from alcs_model.playoffs import _pick_starter, bracket, build_state
from synth import make_league

AL = ["TB", "CWS", "CLE", "NYY"]
NL = ["HOU", "BOS", "LAD", "ATL"]


def _team(t, lg, w, div, wc):
    return {"team": t, "team_id": 0, "name": t, "league": lg, "w": w, "l": 162 - w, "pct": w / 162,
            "division_rank": div, "wildcard_rank": wc}


def _game(pk, d, gt, desc, n, home, away, final=None, ph=None, pa=None, best=5):
    def side(t, sc, prob):
        real = t in AL + NL
        return {"team": t if real else None, "team_id": 0, "name": t, "score": sc, "probable": prob}
    return {"game_pk": pk, "date": d, "game_type": gt, "description": f"{desc} Game {n}", "series": desc,
            "game_number": n, "games_in_series": best, "status": "Final" if final else "Scheduled",
            "abstract": "Final" if final else "Preview", "venue": None,
            "home": side(home, final[1] if final else None, ph), "away": side(away, final[0] if final else None, pa)}


@pytest.fixture(scope="module")
def setup():
    return build_fixture()


def build_fixture():
    """Eight-team bracket in the Division Series: TB up 2-0, CLE and LAD split, BOS up 2-0 on HOU."""
    raw = make_league(n_days=120, teams=AL + NL)
    p = prepare_pitches(raw)
    pa = plate_appearances(p)
    start = p["game_date"].max().date() + timedelta(days=3)
    teams = [_team("TB", "AL", 98, "1", None), _team("CLE", "AL", 90, "1", None),
             _team("NYY", "AL", 92, "2", "1"), _team("CWS", "AL", 85, "2", "3"),
             _team("LAD", "NL", 96, "1", None), _team("HOU", "NL", 91, "1", None),
             _team("ATL", "NL", 89, "2", "1"), _team("BOS", "NL", 86, "3", "2")]
    games, pk = [], 9_000_000
    d = lambda k: str(start + timedelta(days=k))  # noqa: E731
    for desc, hi, lo, res in (("ALDS 'A'", "TB", "NYY", [(0, 1), (2, 5)]), ("ALDS 'B'", "CLE", "CWS", [(3, 0), (4, 3)]),
                              ("NLDS 'A'", "LAD", "ATL", [(3, 5), (3, 2)]), ("NLDS 'B'", "HOU", "BOS", [(2, 3), (3, 4)])):
        for n, (h, a) in enumerate([(hi, lo), (hi, lo), (lo, hi), (lo, hi), (hi, lo)], 1):
            pk += 1
            games.append(_game(pk, d(n), "D", desc, n, h, a, res[n - 1] if n <= 2 else None))
    for lg, off in (("AL", 8), ("NL", 7)):
        for n, h in enumerate(["Higher", "Higher", "Lower", "Lower", "Lower", "Higher", "Higher"], 1):
            pk += 1
            o = "Lower" if h == "Higher" else "Higher"
            games.append(_game(pk, d(off + n), "L", f"{lg}CS", n, f"{lg} {h} Seed", f"{lg} {o} Seed", best=7))
    for n, h in enumerate(["Higher", "Higher", "Lower", "Lower", "Lower", "Higher", "Higher"], 1):
        pk += 1
        games.append(_game(pk, d(18 + n), "W", "World Series", n, f"WS {h} Seed", "WS Other", best=7))
    return p, pa, {"teams": teams, "games": games}


def test_state(setup):
    _, _, br = setup
    st = build_state(br)
    assert st["seeds"]["TB"] == 1 and st["seeds"]["CLE"] == 2 and st["seeds"]["NYY"] == 3
    ds = [s for s in st["series"] if s["round"] == "DS"]
    assert len(ds) == 4 and all(s["need"] == 3 for s in ds)
    tb = next(s for s in ds if "TB" in s["known"])
    assert tb["wins"] == {"TB": 2} and tb["teams"] == ["TB", "NYY"] and tb["label"] == "AL Division Series"
    lcs = [s for s in st["series"] if s["round"] == "LCS"]
    assert sorted(s["league"] for s in lcs) == ["AL", "NL"] and all(len(s["games"]) == 7 for s in lcs)


def test_bracket_sim(setup, cfg):
    p, pa, br = setup
    res = bracket(cfg, p, pa, br, [], None, n=200, seed=1, verbose=False)
    t = res["teams"]
    assert abs(sum(v["p_title"] for v in t.values()) - 1) < 1e-9
    assert abs(sum(v["p_pennant"] for v in t.values() if v["league"] == "AL") - 1) < 1e-9
    for s in res["series"]:
        if s["round"] == "DS":
            assert abs(sum(s["p_win"].values()) - 1) < 1e-9
    assert t["TB"]["p_ds"] > 0.5            # up 2-0 in a best of five
    assert len(res["next_games"]) == 4 and all(0 < g["p_home"] < 1 for g in res["next_games"])


def test_pick_starter_respects_rest():
    d = date(2026, 10, 10)
    last = {1: d - timedelta(days=2), 2: d - timedelta(days=6), 3: d - timedelta(days=5)}
    assert _pick_starter([1, 2, 3], last, d) == 2


def test_wild_card_feeds_division_series(setup, cfg):
    """Before the Wild Card round ends, a bye seed's opponent comes from the right Wild Card series."""
    p, pa, _ = setup
    start = p["game_date"].max().date() + timedelta(days=3)
    d = lambda k: str(start + timedelta(days=k))  # noqa: E731
    teams = [_team("TB", "AL", 98, "1", None), _team("CLE", "AL", 94, "1", None), _team("NYY", "AL", 92, "1", None),
             _team("CWS", "AL", 90, "2", "1"), _team("HOU", "AL", 88, "2", "2"), _team("BOS", "AL", 86, "3", "3")]
    AL6 = ["TB", "CLE", "NYY", "CWS", "HOU", "BOS"]
    games, pk = [], 9_500_000

    def g(gt, desc, n, home, away, final=None, best=3):
        nonlocal pk
        pk += 1
        x = _game(pk, d(n), gt, desc, n, home, away, final, best=best)
        for s_ in ("home", "away"):
            x[s_]["team"] = x[s_]["name"] if x[s_]["name"] in AL6 else None
        return x
    # WC: 4 CWS vs 5 HOU (CWS up 1-0), 3 NYY vs 6 BOS (not started)
    games += [g("F", "AL Wild Card", 1, "CWS", "HOU", (2, 5)), g("F", "AL Wild Card", 2, "CWS", "HOU"),
              g("F", "AL Wild Card", 3, "CWS", "HOU")]
    games += [g("F", "AL Wild Card", n, "NYY", "BOS") for n in (1, 2, 3)]
    for n, h in enumerate(["TB", "TB", "W", "W", "TB"], 1):
        games.append(g("D", "ALDS", 3 + n, h, "W" if h == "TB" else "TB", best=5))
        games[-1]["home" if h == "W" else "away"]["name"] = "AL Wild Card 'A' Winner"
    for n, h in enumerate(["CLE", "CLE", "W", "W", "CLE"], 1):
        games.append(g("D", "ALDS", 3 + n, h, "W" if h == "CLE" else "CLE", best=5))
        games[-1]["home" if h == "W" else "away"]["name"] = "AL Wild Card 'B' Winner"
    br = {"teams": teams, "games": games}
    st = build_state(br)
    from alcs_model.playoffs import feeder
    ds = {s["known"][0]: s for s in st["series"] if s["round"] == "DS"}
    wc = {s["key"]: s for s in st["series"] if s["round"] == "WC"}
    assert set(wc[feeder(ds["TB"], st)]["known"]) == {"CWS", "HOU"}
    assert set(wc[feeder(ds["CLE"], st)]["known"]) == {"NYY", "BOS"}
    res = bracket(cfg, p, pa, br, [], None, n=150, seed=2, verbose=False)
    t = res["teams"]
    assert set(t) == set(AL6)
    assert t["TB"]["p_wc"] is None and 0 < t["CWS"]["p_wc"] < 1
    ds_tb = next(s for s in res["series"] if s["round"] == "DS" and "TB" in s["known"])
    assert set(ds_tb["p_win"]) <= {"TB", "CWS", "HOU"} and abs(sum(ds_tb["p_win"].values()) - 1) < 1e-9
    assert abs(t["CWS"]["p_wc"] + t["HOU"]["p_wc"] - 1) < 1e-9
    assert abs(sum(v["p_ds"] for v in t.values()) - 2) < 1e-9


def test_render_hub(setup, cfg, tmp_path, monkeypatch, capsys):
    import alcs_model.playoffs as po
    p, pa, br = setup
    res = bracket(cfg, p, pa, br, [], None, n=60, seed=3, verbose=False)
    monkeypatch.setattr(po, "DOCS", tmp_path)
    monkeypatch.setattr(po, "OUTPUT", tmp_path)
    po.render(res)
    html = (tmp_path / "index.html").read_text()
    assert html.startswith("<!doctype html>") and "__DATA__" not in html and '"next_games"' in html
    assert "—" not in html  # house style: no em dashes
    po._report(res)
    assert "title odds" in capsys.readouterr().out
