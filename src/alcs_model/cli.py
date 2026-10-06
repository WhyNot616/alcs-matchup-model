"""Command line interface.

    alcs pull [--scope league|teams] [--refresh]   download Statcast, leaderboards and schedule
    alcs backtest                                  test whether the matchup model predicts anything
    alcs build [--n 5000] [--bootstrap 20]         fit, simulate, write output/ and docs/index.html
    alcs render                                    rebuild docs/index.html from output/dashboard_data.json
    alcs all                                       pull (incremental) + build

Run `python -m alcs_model <command>` if the `alcs` script is not on your PATH.
"""
from __future__ import annotations

import argparse
import json

from .config import OUTPUT, load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="alcs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to a series YAML (default config/series.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_pull = sub.add_parser("pull", help="download data")
    p_pull.add_argument("--scope", choices=["league", "teams"], default=None)
    p_pull.add_argument("--refresh", action="store_true", help="re-download everything instead of only new dates")
    sub.add_parser("backtest", help="run the backtest only")
    for name in ("build", "all"):
        sp = sub.add_parser(name, help="fit + simulate + write dashboard" if name == "build" else "pull + build")
        sp.add_argument("--n", type=int, default=None, help="number of simulated series")
        sp.add_argument("--bootstrap", type=int, default=None, help="bootstrap resamples for the uncertainty range")
        sp.add_argument("--skip-backtest", action="store_true", help="reuse output/backtest.json")
        sp.add_argument("--seed", type=int, default=None)
        if name == "all":
            sp.add_argument("--scope", choices=["league", "teams"], default=None)
    sub.add_parser("render", help="rebuild docs/index.html from saved data")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)

    if a.cmd == "pull":
        from .data import pull_all
        pull_all(cfg, scope=a.scope, refresh=a.refresh)
    elif a.cmd == "backtest":
        from .pa_model import backtest
        from .pipeline import load_prepared
        p, pa = load_prepared(cfg)
        res = backtest(p, pa, cfg.model)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "backtest.json").write_text(json.dumps(res, indent=2))
    elif a.cmd in ("build", "all"):
        if a.cmd == "all":
            from .data import pull_all
            pull_all(cfg, scope=getattr(a, "scope", None))
        from .pipeline import run
        run(cfg, n_series=a.n, n_boot=a.bootstrap, skip_backtest=a.skip_backtest, seed=a.seed)
    elif a.cmd == "render":
        from .pipeline import render
        render()


if __name__ == "__main__":
    main()
