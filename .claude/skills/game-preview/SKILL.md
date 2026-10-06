---
name: game-preview
description: Write a short, data-backed preview of an upcoming playoff game (or the next game in a series) from the bracket output - starters, model vs market, what drives the number, and how much to trust it. Use when Ami asks about a specific upcoming game.
argument-hint: "<team or series, e.g. 'TB' or 'NLDS LAD ATL'>"
---

# Game preview

Game: $ARGUMENTS

Sources, all in the repo (read, do not guess):
- `output/bracket.json`: `next_games` (starters, announced flags, p_home, exp_runs, market, market_total,
  market_ml), `series` (status, p_win, lengths), `teams` (rotation, lineups), `names`.
- `output/dashboard_data.json`: pitcher cards with Stuff+, bullpen tendencies, only if the game involves
  the two deep-dive teams in `config/series.yaml`.
- `output/game_backtest.json` and `NOTES.md`: how the model has done vs baselines and the market.

If the data is older than today's date or the starters look stale, say so and offer to run `/refresh`.

Write it in Ami's register: plain, direct sentences, no em dashes, no hype. Cover:
1. Matchup, date, series state, starters (say which are announced and which are projected).
2. Model win probability and expected runs, next to the market's no-vig probability and total. State the
   gap in points.
3. What drives the model's number (starter quality and Stuff+, home field, lineup vs the starter's
   hand, bullpen rest). Keep it to things the data actually shows.
4. Trust level, from the backtest: over 777 late-season games the market beat the model and the model's
   disagreements carried no detectable information. A gap is a prompt to check inputs, not a bet.
Keep it under 200 words unless she asks for more.
