---
name: model-check
description: Run the tests and summarize how the model is doing - PA backtest, game backtest vs baselines and the betting market, calibration, October scoring - and flag anything that regressed. Use after any modeling change or when Ami asks whether the model is any good.
allowed-tools: Bash(ALCS_OFFLINE=1 pytest *) Bash(ALCS_OFFLINE=1 pytest) Bash(git diff *)
---

# Model check

1. `ALCS_OFFLINE=1 pytest -q`. If anything fails, stop and fix or report that first.
2. Read `output/backtest.json`, `output/game_backtest.json`, `output/postseason_env.json` and compare
   with the numbers in `NOTES.md`.
3. Report bluntly (Ami wants critical feedback, no softening):
   - PA log loss vs league; which of pitch-mix and stuff are on and why (validation, not test)
   - late-season game Brier for model, Pythagorean log5, win% log5, home field, market, with the
     model-minus-baseline intervals; say plainly if the model is not better
   - market regression weight and standard error, and what it means
   - favorites' stated vs actual win rate
   - playoff games so far, labeled as anecdote while n is small
4. If a change made something worse on validation, recommend reverting it.
5. Update the "What the backtests say" section of `NOTES.md` if the numbers changed.
Never tune anything on the test window, and never feed market lines into the model.
