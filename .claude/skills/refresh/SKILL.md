---
name: refresh
description: Pull the latest data, re-simulate the bracket (and optionally the full model and backtests), summarize what changed, and publish the pages. Use when Ami asks to update, refresh or rerun the playoff model.
argument-hint: "[bracket|full]"
arguments: [mode]
disable-model-invocation: true
allowed-tools: Bash(python -m alcs_model *) Bash(git status *) Bash(git diff *) Bash(git log *)
---

# Refresh the postseason model

Mode: `$mode` (default `bracket` if empty).

Before:
- Odds now: !`python -c "import json;b=json.load(open('output/bracket.json'));T=b['teams'];print(b['generated'], {t:round(T[t]['p_title'],3) for t in sorted(T,key=lambda t:-T[t]['p_title'])})" 2>/dev/null || echo "no bracket yet"`

Steps:
1. `git pull --rebase` so local output matches what the scheduled GitHub run committed.
2. `python -m alcs_model pull` (incremental; the first league pull on a new machine takes about
   30 minutes, so warn Ami before starting it if `data/raw/` is empty).
3. If mode is `full`: `python -m alcs_model history`, `python -m alcs_model backtest`,
   `python -m alcs_model backtest-games`, then `python -m alcs_model build --skip-backtest`.
   The full run takes over an hour locally; suggest the GitHub Action (`gh workflow run refresh.yml`)
   instead and only run locally if she says so.
4. `python -m alcs_model playoffs` (about 10 to 15 minutes at 5,000 simulations; `--n 1000` for a quick look).
5. Report, in plain sentences with no em dashes:
   - title and pennant odds that moved by 3 points or more since the "Before" line, and why
     (a game result, an announced starter, a lineup change)
   - each next game: starters (announced or projected), model vs market, gap in points; flag gaps of
     5 points or more as "check the inputs", never as a bet
   - anything that looks wrong: a projected starter who is hurt or was just used, a lineup with a
     player who is out, a team missing from the bracket
6. Ask before committing. If yes: `git add docs/ output/bracket.json output/*.json` (never `data/raw/`),
   commit, `git pull --rebase -X theirs origin main`, push.
