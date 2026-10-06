---
name: new-series
description: Point the two-team deep dive (docs/series.html) at a different series, such as the actual ALCS, NLCS or World Series once the matchup is set. Use when Ami asks for a deep dive on a specific matchup.
argument-hint: "<TEAM1> <TEAM2>"
arguments: [team1, team2]
disable-model-invocation: true
---

# New series deep dive: $team1 vs $team2

1. Confirm both teams are alive and paired in `output/bracket.json` (`series` with matching `teams`, or
   a likely pairing). If the series is not set yet, say how likely it is and ask whether to proceed.
2. Edit `config/series.yaml` (keep a copy of the old values in the commit message, not the file):
   - `series_name`, `teams` (abbreviation, name, venue, roof), `higher_seed` (seed within a league,
     better record in the World Series)
   - `schedule`: dates and home team per game from the bracket's `series[].games`; best-of-7 is 2-3-2
   - `rotation`: start from `teams.<TEAM>.rotation` in `output/bracket.json`, in order, cycling for
     later games; ask Ami about openers or bulk pitchers
   - `lineups`: `teams.<TEAM>.lineups` from the bracket (vs_R, vs_L)
   - `bullpen`: `auto`, carrying `exclude_pitchers` from `config/playoffs.yaml`
   Use `python -m alcs_model who "<name>"` for any id you need. Never guess an id.
3. Build: `python -m alcs_model build --skip-backtest --n 5000 --bootstrap 20`
   (quick check first with `--n 300 --bootstrap 0`). Then `python -m alcs_model render` so the hub links
   to the new page.
4. Open `docs/series.html` and check the summary reads correctly for the new teams; the copy is generated
   from data, so if a sentence names the wrong team, fix the generator in `dashboard/template.html`,
   not the text.
5. Ask before committing and pushing.
