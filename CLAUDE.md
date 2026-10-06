# CLAUDE.md

Guide for Claude Code working in this repository.

## Project

Python package `alcs_model` (src layout) that pulls Statcast data, fits a plate-appearance outcome model,
measures bullpen usage, simulates games PA by PA, and renders two pages: the postseason hub
(`docs/index.html`, whole bracket) and a two-team series deep dive (`docs/series.html`).
Owner: Ami Argentar (Statistics, UIUC). Config-driven: `config/series.yaml` holds teams, schedule,
rotations, lineups, bullpen overrides and model parameters.

Read `NOTES.md` first: current status, what the backtests found, decisions, and known limitations.
Update it when a refresh or experiment changes those findings.

@NOTES.md

## Working with Ami

- Blunt, critical feedback on the model. If something does not beat the baselines or the market, say so.
- Ask before pushing, dispatching the GitHub workflow, or starting the 30-minute first data pull.
- Project skills: `/refresh [bracket|full]`, `/pin <team> <change>`, `/game-preview <team>`,
  `/model-check`, `/new-series <T1> <T2>` (in `.claude/skills/`).

## Commands

- Install: `pip install -e ".[dev]"` (Python 3.10+)
- Tests: `ALCS_OFFLINE=1 pytest` (synthetic data in `tests/synth.py`, no network)
- Pull data: `python -m alcs_model pull [--scope league|teams] [--refresh]`
- Build the deep dive: `python -m alcs_model build [--n 5000] [--bootstrap 20] [--skip-backtest]`
- Simulate the bracket and write the hub: `python -m alcs_model playoffs [--n 5000]`
- Quick build while iterating: `python -m alcs_model build --n 300 --bootstrap 0 --skip-backtest`
- Re-render the HTML only: `python -m alcs_model render`
- Find a player's MLBAM id: `python -m alcs_model who "Rasmussen"`
- Game-level backtest: `python -m alcs_model backtest-games [--no-regular] [--post-sims 4000] [--reg-sims 300]`
- Postseason run environment: `python -m alcs_model history` (writes `output/postseason_env.json`)

## Layout

- `src/alcs_model/data.py`: Savant/MLB pulls and caches (`data/raw/`, gitignored)
- `features.py`: `prepare_pitches`, `plate_appearances`, xRV, leverage table, active rosters
- `matchups.py`: pitch-type profiles with two-level shrinkage, `pair_edges`, `matchup_detail`
- `pa_model.py`: outcome rates, `matchup_probs` (odds ratio + pitch-mix tilt + park), `backtest`
- `bullpen.py`: `appearances`, `reliever_tendencies`, `situational_splits`, `starter_hooks`, `UsageModel`
- `simulate.py`: `sim_game`, `sim_series`, baserunning in `_advance`, fatigue across dates
- `aggregates.py`: raw descriptive tables for the dashboard (field order in `FIELDS`)
- `backtest_games.py`: `World` (model fit on a training window), `game_backtest`, `score_games`; writes
  `output/game_backtest.json`, which `render()` merges into the dashboard
- `postseason_env.py`: actual vs expected scoring in past postseasons; `load_factor` feeds the simulator
- `market.py`: matches ESPN/DraftKings lines to games, `model_vs_market` regression, `totals_check`
- `stuff.py`: pitch-quality model and Stuff+; `pa_model.stuff_setting(bt)` returns the mode chosen on validation
- `pipeline.py`: `run()` builds the deep dive, writes `output/dashboard_data.json` + `docs/series.html`
- `playoffs.py`: `build_state` (series grouped by round + team pair; LCS/WS by league), `feeder` (WC -> DS),
  `simulate_bracket`, `bracket()`, `run()`; writes `output/bracket.json` + `docs/index.html`.
  Optional overrides in `config/playoffs.yaml`
- `dashboard/template.html` (deep dive) and `dashboard/playoffs.html` (hub); `__DATA__` is replaced with JSON

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
- The game backtest must stay leak-free: anything about a game (lineups, starters, bullpen, fatigue) may
  only use data dated before that game, and models are fit on the training window only.
- Stuff, pitch-mix and shrinkage settings are chosen on the PA backtest's validation window, never its test window.
- Market lines are a benchmark only. Never feed them into the model's own predictions, or the
  comparison stops meaning anything.
