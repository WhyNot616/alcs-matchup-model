# ALCS Matchup Model

A pitch-level scouting model and series simulator for the 2026 ALCS between the White Sox and Rays. It
pulls every Statcast pitch of the season from Baseball Savant, builds hitter and pitcher profiles by
pitch type, measures how each manager uses his bullpen, and simulates the series plate appearance by
plate appearance. The output is an interactive dashboard (`docs/index.html`, published with GitHub Pages).

Everything is driven by one config file, so the same code works for any two teams and any series.

## What it does

| Step | Module | What happens |
|---|---|---|
| Data | `data.py` | Pulls Statcast pitch data (league-wide or just the two teams), Savant leaderboards (OAA, sprint speed, framing, baserunning, park factors) and the MLB schedule. Caches to `data/raw/` and refreshes incrementally. |
| Features | `features.py` | Cleans types, tags swings/whiffs/outcomes, builds a plate-appearance table, computes expected run value (xRV) and an empirical leverage index for every game state. |
| Matchups | `matchups.py` | Pitch-type run value profiles for every hitter and pitcher with two-level shrinkage, pitch usage by batter side, and the arsenal edge for any hitter vs pitcher pair. |
| Outcome model | `pa_model.py` | Probabilities of K, BB, 1B, 2B, 3B, HR and outs for any plate appearance (multinomial odds ratio with platoon splits), plus a pitch-mix tilt. Includes a backtest. |
| Bullpens | `bullpen.py` | Every relief appearance with entry inning, score, leverage, rest and results. Roles, situational splits, starter hook patterns, and a usage model of which reliever each manager calls in each situation. |
| Simulation | `simulate.py` | Plays games PA by PA: outcome draws, baserunning, starter hooks, bullpen choices and fatigue across the series calendar, postseason extra-inning rules. Bootstraps the season for an uncertainty range. |
| Game backtest | `backtest_games.py` | Predicts real games from pregame information only (actual lineups and starters, recent bullpens, real fatigue) and scores them against coin flip, home field and log5 baselines with bootstrap intervals. |
| Report | `pipeline.py`, `aggregates.py` | Runs it all and writes `output/dashboard_data.json` and `docs/index.html`. |

## Setup (VS Code)

```bash
git clone https://github.com/WhyNot616/alcs-matchup-model.git
cd alcs-matchup-model
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Open the folder in VS Code and pick the `.venv` interpreter. The repo includes recommended extensions,
pytest settings and two launch configurations (Run and Debug panel).

## Running it

```bash
alcs pull                 # first run: every 2026 pitch, about 30 minutes; later runs only fetch new days
alcs pull --scope teams   # faster alternative: only games involving the two teams (no backtest)
alcs build                # fit, backtest, simulate 5,000 series + 20 bootstrap reruns, write the dashboard
alcs build --n 500 --bootstrap 0 --skip-backtest   # quick iteration while editing
alcs backtest             # plate-appearance model check
alcs history              # fit the October run environment on the 2021-2025 postseasons
alcs backtest-games       # predict every 2026 playoff game and every game after Aug 1 pregame, score them,
                          # and compare with DraftKings closing lines (via ESPN)
alcs render               # rebuild docs/index.html from the last saved output
alcs all                  # pull new data, then build
```

If the `alcs` command is not found, use `python -m alcs_model <command>`. Open `docs/index.html` in a
browser to see the dashboard. `notebooks/explore.py` has VS Code notebook cells for poking at one
matchup, one bullpen, or the leverage table.

## Updating for the series

Edit `config/series.yaml`:

- **rotation**: starters by game, by MLBAM id (the number in a player's Baseball Savant URL). Use
  `opener_max_bf` and `bulk` for an opener plus bulk pitcher.
- **lineups**: batting orders against right-handed and left-handed starters.
- **bullpen**: `auto` picks pitchers active in the last four weeks. Use `exclude` and `include` to
  match the announced roster (McClanahan is excluded by default after his forearm injury).
- **schedule**: dates drive bullpen rest in the simulator.

Then run `alcs all`.

## Running it on GitHub

`.github/workflows/refresh.yml` runs the full pipeline on GitHub's servers and commits the updated
dashboard. Start it from the **Actions** tab with "Run workflow", or let it run every morning in
October. `.github/workflows/ci.yml` runs the tests on every push. Turn on GitHub Pages (Settings,
Pages, deploy from branch `main`, folder `/docs`) to get a public link to the dashboard.

## Working with Claude Code

`CLAUDE.md` describes the project layout, commands and conventions so Claude Code can work in the
repo from the terminal (`claude` in the project folder). Useful asks: "update the rotation for Game 3
and rebuild", "add a split for pitches with runners in scoring position", "why does the backtest say
the pitch-mix tilt does not help?".

## How to read the model check

Single plate appearances are mostly noise, so even good baseball models only beat league averages by
a percent or two in log loss. The backtest trains on games before `backtest_split` and scores the
rest. If the pitch-mix tilt does not lower out-of-sample log loss, the pipeline sets it to zero and the
pitch-mix tables become descriptive only. The dashboard says which way it went.

## Data and terms

Statcast and MLB data come from Baseball Savant and the MLB Stats API and remain subject to MLB
Advanced Media's terms. Raw data is not committed to this repository; the pipeline downloads it to
`data/raw/`. Please keep request volume reasonable.

## Tests

```bash
pytest
```

Tests run on a synthetic season with the same columns as Statcast, so they need no network access.
