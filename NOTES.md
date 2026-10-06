# Project notes

Working notes for anyone (or any Claude Code session) picking this project up. Update the status and
findings when a refresh or an experiment changes them. Numbers below are from the run of 2026-10-06.

## Status

- The pipeline runs end to end on GitHub Actions (`refresh.yml`): full run every morning in October and
  the first week of November, plus a bracket-only run every afternoon. It commits `docs/` and `output/`.
- `docs/index.html` is the postseason hub (whole bracket). `docs/series.html` is the two-team deep dive,
  still configured for White Sox vs Rays (`config/series.yaml`).
- Bracket on 2026-10-06: AL DS TB 2-0 NYY, CWS 2-0 CLE; NL DS MIL 2-0 SD, LAD 1-1 ATL.
  Title odds: MIL 33%, TB 25%, LAD 21%, CWS 9%, ATL 7%, CLE 2%, SD 2%, NYY 1%.
  The most likely ALCS is TB vs CWS (79% of simulations), so the deep dive config still fits.

## What the backtests say (be honest about these when describing the model)

Plate-appearance model (train before 2026-08-01, 124,562 PA; test after, 58,591 PA):
- Log loss 1.4398 vs 1.4569 for league average, a 1.2% improvement. Shrinkage x2 and a 120-day
  recency half-life were chosen on a validation window.
- Pitch-mix tilt: no gain, lambda is 0. The pitch-mix tables are descriptive only.
- Stuff model: as a tilt, no gain. As each pitcher's prior (regression target), +0.019% on test. Chosen
  on validation, so it is on (`stuff_mode = prior`). Real but tiny.
- Stuff stability (174 pitchers, 400+ pitches each side of Aug 1): early stuff correlates .44 with late
  results, early results .42, both together .48.

Game level (pregame information only, leak-free):
- Late season, 777 games: Brier .2406 model, .2419 Pythagorean log5, .2392 win% log5, .2486 home field,
  .2372 DraftKings close. The model is not measurably better than the simple baselines and is worse
  than the market (difference +.0034, 95% interval -.0013 to +.0083).
- Regression of results on market + (model minus market): weight 0.22, standard error 0.27. The
  model's disagreements with the line carry no detectable information.
- Calibration: favorites given 57.9% won 57.1%. Reasonable.
- Playoffs, 17 games: model .2258, market .2379. Far too few games to mean anything.

October scoring: 2021-2025 postseasons scored 0.903 of what regular-season numbers predict (95%
0.845 to 0.963, 208 games); 2026 so far 0.824 on 17 games. The simulator is calibrated to 0.903.

## Decisions and rules

- Market lines are a benchmark, never a model input.
- Settings are chosen on validation windows, never on test.
- The game backtest only uses information dated before each game.
- Dashboard copy is generated from data in JS, plain sentences, no em dashes.

## Known limitations

- No injury or roster feed. Announced probables and recent lineups are the only signal; use
  `config/playoffs.yaml` to pin rotations, lineups and excluded pitchers.
- Rotation picker: most rested of the last four starters with at least four days of rest. It does not
  know about openers or bullpen games in the bracket view.
- Placeholder games (LCS/WS before teams are known) assume the standard 2-3-2 home pattern by seed.
- Single PAs are mostly noise, so player-level edges in the deep dive are directional, not precise.

## Ideas not yet tried

- Shrink game probabilities toward the Pythagorean baseline (or a 50/50 blend) and test if Brier improves.
- Per-series deep dives generated for every LCS and World Series matchup automatically.
- Starter-specific pitch counts in October (managers pull starters earlier than in the season).
