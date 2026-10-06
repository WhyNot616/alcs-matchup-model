---
name: pin
description: Override the bracket model's automatic choices in config/playoffs.yaml (a team's rotation, its lineup against righties or lefties, or a pitcher to exclude), then re-simulate. Use when a starter is announced, a player is hurt, or a projected lineup or rotation is wrong.
argument-hint: "<team> <what to change, e.g. 'Kay starts G3' or 'Robert out vs RHP' or 'exclude Bednar'>"
---

# Pin a rotation, lineup or exclusion

Request: $ARGUMENTS

1. Read `config/playoffs.yaml` and the team's current inputs in `output/bracket.json`
   (`teams.<TEAM>.rotation`, `teams.<TEAM>.lineups`, names in `names`).
2. Find every player's MLBAM id with `python -m alcs_model who "<name>"`. If a name matches more than
   one player, ask which one. Never guess an id.
3. Edit `config/playoffs.yaml`:
   - `rotation: {TEAM: [ids in order]}`: used after MLB's announced probables run out. Announced
     probables always win, so pinning a starter MLB has already announced changes nothing.
   - `lineups: {TEAM: {vs_R: [9 ids], vs_L: [9 ids]}}`: full batting order, exactly nine ids.
   - `exclude_pitchers: [ids]`: off the roster or hurt; removes them from every bullpen.
   Keep comments with each id's name so the file stays readable.
4. Re-simulate: `python -m alcs_model playoffs --n 2000` (fast) and report how the affected series and
   title odds moved, in one or two plain sentences.
5. Remind Ami the override stays until she removes it, and to clear it once MLB's data catches up.
