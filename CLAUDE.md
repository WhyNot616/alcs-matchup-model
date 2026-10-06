# CLAUDE.md

Guide for Claude Code working in this repository.

## Project

Python package `alcs_model` (src layout) that pulls Statcast data, fits a plate-appearance outcome model,
measures bullpen usage, simulates a best-of-seven series PA by PA, and renders an HTML dashboard.
Owner: Ami Argentar (Statistics, UIUC). Config-driven: `config/series.yaml` holds teams, schedule,
rotations, lineups, bullpen overrides and model parameters.

## Commands

- Install: `pip install -e ".[dev]"` (Python 3.10+)
- Tests: `ALCS_OFFLINE=1 pytest` (synthetic data in `tests/synth.py`, no network)
- Pull data: `python -m alcs_model pull [--scope league|teams] [--refresh]`
- Build: `python -m alcs_model build [--n 5000] [--bootstrap 20] [--skip-backtest]`
- Quick build while iterating: `python -m alcs_model build --n 300 --bootstrap 0 --skip-backtest`
- Re-render the HTML only: `python -m alcs_model render`

## Layout

- `src/alcs_model/data.py`: Savant/MLB pulls and caches (`data/raw/`, gitignored)
- `features.py`: `prepare_pitches`, `plate_appearances`, xRV, leverage table, active rosters
- `matchups.py`: pitch-type profiles with two-level shrinkage, `pair_edges`, `matchup_detail`
- `pa_model.py`: outcome rates, `matchup_probs` (odds ratio + pitch-mix tilt + park), `backtest`
- `bullpen.py`: `appearances`, `reliever_tendencies`, `situational_splits`, `starter_hooks`, `UsageModel`
- `simulate.py`: `sim_game`, `sim_series`, baserunning in `_advance`, fatigue across dates
- `aggregates.py`: raw descriptive tables for the dashboard (field order in `FIELDS`)
- `pipeline.py`: `run()` orchestrates everything and writes `output/dashboard_data.json` + `docs/index.html`
- `dashboard/template.html`: the dashboard; `__DATA__` is replaced with the JSON at render time

## Conventions

- Outcomes are always ordered `["K", "BB", "1B", "2B", "3B", "HR", "OUT"]` (`features.OUTCOMES`).
- Run values: hitter view is positive for the offense; pitcher view flips the sign. Profiles in
  `matchups.py` are relative to league average for each pitch type; aggregates are raw season values.
- Player ids are MLBAM ints. Names are "Last, First".
- Keep pandas code compatible with pandas 2.1+ and 3.x (no `groupby.apply(include_groups=...)`,
  avoid chained assignment, groupers must match frame length).
- Dashboard copy: plain, direct sentences and no em dashes. Text in the dashboard is generated from
  the data in JS so it stays correct after a refresh; do not hard-code numbers in the template.
- Do not commit anything in `data/raw/`.

## Gotchas

- Savant caps each CSV at 25,000 rows; `_statcast_window` splits date ranges recursively.
- A league-wide pull is about 750k pitches and takes ~30 minutes the first time; later pulls are incremental.
- The backtest needs league scope; with `teams` scope one side of most PAs has a tiny sample.
- `home_pa_tilt` was calibrated so identical teams give the home side about 53.5%; recheck if the
  baserunning rules in `_advance` change.
