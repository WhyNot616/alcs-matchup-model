# %% [markdown]
# # Explore the ALCS matchup model
# Run cells with "Run Cell" in VS Code (Python + Jupyter extensions). Run `alcs pull` first so the
# Statcast cache exists in data/raw/.

# %%
import pandas as pd

from alcs_model.config import load_config
from alcs_model.features import plate_appearances, prepare_pitches
from alcs_model.data import load_statcast
from alcs_model.pipeline import fit, rosters, bats_throws
from alcs_model.pa_model import matchup_probs, expected_woba, batter_side
from alcs_model.matchups import matchup_detail, pair_edges
from alcs_model.bullpen import appearances, reliever_tendencies

cfg = load_config()
pitches = prepare_pitches(load_statcast(cfg))
pa = plate_appearances(pitches)
F = fit(cfg, pitches, pa)
ros = rosters(cfg, pitches)
bats, throws = bats_throws(pitches)
print(f"{len(pitches):,} pitches, {len(pa):,} PA, data through {F.as_of.date()}")

# %% [markdown]
# ## One matchup in detail
# Change the ids to any hitter / pitcher (MLBAM ids from Baseball Savant URLs).

# %%
hitter, pitcher = 808959, 656876   # Munetaka Murakami vs Drew Rasmussen
hand = throws[pitcher]
side = batter_side(bats[hitter], hand)
edge = float(pair_edges(F.profiles, pd.Series([hitter]).to_numpy(), pd.Series([pitcher]).to_numpy(),
                        pd.Series([side]).to_numpy())[0])
probs = matchup_probs(F.rates, hitter, side, pitcher, hand, edge, lam=0.0)
print("expected wOBA:", round(expected_woba(probs), 3), "| arsenal edge:", round(edge, 2))
pd.DataFrame(matchup_detail(F.profiles, hitter, pitcher, side))

# %% [markdown]
# ## Bullpen tendencies for one team

# %%
team = "TB"
app = appearances(pitches, team, F.li)
reliever_tendencies(app[app["game_type"].astype(str).eq("R")], {}, ros[team]["bullpen"]).round(3)

# %% [markdown]
# ## Leverage by inning and score (home team view, bases empty, no outs)

# %%
li = F.li
li[(li["outs_when_up"] == 0) & (li["base_state"] == 0)].pivot_table(
    index="inn", columns="hsd", values="li", aggfunc="mean").round(2)
