# Marcos:
# - ['marcos_1']: Opponent strength (defensive and offensive rating of the opponent, with everything amplified in the playoffs)
# - ['marcos_2']: Away vs home game
# - ['marcos_3']: Shot clock pressure
# - ['marcos_4']: Whats on the line (playoffs wise → farther-in games in a series worth more)

#Load the dataframes in and make the four features below:

#...

# Making one combined dataframe
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

reg = pd.read_csv("datasets/reg_season_pbp_since_2020_21.csv", encoding="latin1")
po  = pd.read_csv("datasets/playoffs_pbp_since_2020_21.csv", encoding="latin1")
stats = pd.read_csv("datasets/clutch_player_stats_since_2020_21.csv", encoding="latin1")


reg["is_playoffs"] = 0
po["is_playoffs"] = 1

pbp_cols = [
    "actionNumber",
    "clock",
    "period",
    "actionType",
    "subType",
    "personId",
    "teamId",
    "teamTricode",
    "gameId",
    "scoreHome",
    "scoreAway",
    "orderNumber",
    "isFieldGoal",
    "is_playoffs",
    "shotResult",
    "shotDistance",
    "pointsTotal",
    "turnoverTotal",
    "reboundTotal",
    "assistPersonId",
    "stealPersonId",
    "blockPersonId"
]

reg = reg[[c for c in pbp_cols if c in reg.columns]]
po  = po[[c for c in pbp_cols if c in po.columns]]

pbp_combined = pd.concat([reg, po], ignore_index=True)

pbp_combined.to_csv("pbp_combined_slim.csv", index=False)


#Feature 5(opponent strength): average +/– (PLUS_MINUS) of the opponent team in the clutch while player is in

pbp = pd.read_csv("pbp_combined_slim.csv", encoding="latin1", engine="python")
stats = pd.read_csv("datasets/clutch_player_stats_since_2020_21.csv", encoding="latin1", engine="python")

#compute team strength
team_strength = (
    stats.groupby("TEAM_ID", as_index=False)
         .agg(team_strength=("PLUS_MINUS", "mean"))
)

#make ints
team_strength["TEAM_ID"] = pd.to_numeric(team_strength["TEAM_ID"], errors="coerce")
team_strength = team_strength.dropna(subset=["TEAM_ID"])
team_strength["TEAM_ID"] = team_strength["TEAM_ID"].astype(int)

stats["TEAM_ID"] = pd.to_numeric(stats["TEAM_ID"], errors="coerce")
stats = stats.dropna(subset=["TEAM_ID"])
stats["TEAM_ID"] = stats["TEAM_ID"].astype(int)


pbp["TEAM_ID"] = pd.to_numeric(pbp["teamId"], errors="coerce")
pbp = pbp.dropna(subset=["TEAM_ID"])
pbp["TEAM_ID"] = pbp["TEAM_ID"].astype(int)


#find teams in each game
game_to_teams = (
    pbp.groupby("gameId")["TEAM_ID"]
       .unique()
       .apply(list)
)

opp_map = {}
for game, teams in game_to_teams.items():
    if len(teams) == 2:  # only valid games
        t1, t2 = teams
        opp_map[(game, t1)] = t2
        opp_map[(game, t2)] = t1


#make team column
pbp["OPP_TEAM_ID"] = pbp.apply(
    lambda r: opp_map.get((r["gameId"], r["TEAM_ID"])),
    axis=1
)

pbp["OPP_TEAM_ID"] = pd.to_numeric(pbp["OPP_TEAM_ID"], errors="coerce")
pbp = pbp.dropna(subset=["OPP_TEAM_ID"])
pbp["OPP_TEAM_ID"] = pbp["OPP_TEAM_ID"].astype(int)


#merge team strength
pbp = pbp.merge(
    team_strength,
    left_on="OPP_TEAM_ID",
    right_on="TEAM_ID",
    how="left",
    suffixes=("", "_opp")
)

pbp = pbp.rename(columns={"team_strength": "opponent_strength"})


#find per player
feature5 = (
    pbp.groupby("personId", as_index=False)
       .agg(marcos_1=("opponent_strength", "mean"))
       .rename(columns={"personId": "PLAYER_ID"})
)

feature5["PLAYER_ID"] = pd.to_numeric(feature5["PLAYER_ID"], errors="coerce")
feature5 = feature5.dropna(subset=["PLAYER_ID"])
feature5["PLAYER_ID"] = feature5["PLAYER_ID"].astype(int)


#merge to stats
stats_with_feature5 = stats.merge(feature5, on="PLAYER_ID", how="left")


#find average for each player along every season
final_stats = (
    stats_with_feature5
    .groupby(["PLAYER_ID", "PLAYER_NAME"], as_index=False)
    .agg(marcos_1=("marcos_1", "mean"))
)


final_stats.to_csv("stats_with_feature5_one_row_per_player.csv", index=False)

print(final_stats.head())
print(final_stats.shape)


#Feature 6 (Away vs Home)


import pandas as pd
import numpy as np
import re

# =========================================
# Helper: parse clock like "PT01M23.00S" → seconds remaining in period
# =========================================
def clock_to_seconds(clock_str):
    if pd.isna(clock_str):
        return np.nan
    clock_str = str(clock_str)
    m = re.match(r"PT(\d{1,2})M(\d{2})", clock_str)
    if not m:
        return np.nan
    mins = int(m.group(1))
    secs = int(m.group(2))
    return mins * 60 + secs

# =========================================
# 1. Load data
# =========================================
pbp = pd.read_csv("pbp_combined_slim.csv", encoding="latin1", engine="python")
stats = pd.read_csv("datasets/clutch_player_stats_since_2020_21.csv", encoding="latin1", engine="python")

print("Loaded PBP:", pbp.shape)
print("Loaded Stats:", stats.shape)

# Make sure TEAM_IDs in stats are numeric
stats["TEAM_ID"] = pd.to_numeric(stats["TEAM_ID"], errors="coerce")
stats = stats.dropna(subset=["TEAM_ID"])
stats["TEAM_ID"] = stats["TEAM_ID"].astype(int)

# =========================================
# 2. Clean team IDs in PBP (teamId → TEAM_ID)
# =========================================
pbp["TEAM_ID"] = pd.to_numeric(pbp["teamId"], errors="coerce")
pbp = pbp.dropna(subset=["TEAM_ID"])
pbp["TEAM_ID"] = pbp["TEAM_ID"].astype(int)

# =========================================
# 3. Build home team mapping per game
#    Logic: use first scoring event to infer which TEAM_ID is home
# =========================================
home_team_map = {}

for game_id, game_df in pbp.groupby("gameId"):
    # scoring events with positive points
    scoring = game_df[(game_df["pointsTotal"].notna()) & (game_df["pointsTotal"] > 0)]
    if scoring.empty:
        continue

    scoring = scoring.sort_values(["period", "orderNumber"])
    first = scoring.iloc[0]

    teams_in_game = game_df["TEAM_ID"].dropna().unique()
    if len(teams_in_game) < 2:
        continue

    scoring_team = first["TEAM_ID"]
    other_teams = [t for t in teams_in_game if t != scoring_team]
    if not other_teams:
        continue
    other_team = other_teams[0]

    # If first scoring event results in scoreHome > scoreAway,
    # then that scoring team is home. Otherwise, the other is home.
    if first["scoreHome"] > first["scoreAway"]:
        home_team = scoring_team
    else:
        home_team = other_team

    home_team_map[game_id] = home_team

# Map home team onto each row
pbp["homeTeamId"] = pbp["gameId"].map(home_team_map)
pbp["is_home"] = (pbp["TEAM_ID"] == pbp["homeTeamId"]).astype(int)

# =========================================
# 4. Compute shot-clock remaining & filter to scoring plays
# =========================================
pbp["seconds_left_period"] = pbp["clock"].apply(clock_to_seconds)

# derive approximate shot clock: mod 24
pbp["shot_clock_sec"] = pbp["seconds_left_period"] % 24

# Avoid division by zero / NaN
pbp = pbp.dropna(subset=["shot_clock_sec"])
pbp.loc[pbp["shot_clock_sec"] <= 0, "shot_clock_sec"] = 1.0

# Consider only scoring events (pointsTotal > 0)
pbp_shots = pbp[(pbp["pointsTotal"].notna()) & (pbp["pointsTotal"] > 0)]

print("Scoring events for Feature 6:", pbp_shots.shape)

# =========================================
# 5. Compute pressure score per play
# =========================================
# base score = points / shot_clock_remaining
pbp_shots["base_score"] = pbp_shots["pointsTotal"] / pbp_shots["shot_clock_sec"]

# home/away multiplier
pbp_shots["home_mult"] = np.where(pbp_shots["is_home"] == 1, 1.0, 1.2)

# playoff multiplier
pbp_shots["playoff_mult"] = np.where(pbp_shots["is_playoffs"] == 1, 1.5, 1.0)

# final pressure score
pbp_shots["pressure_score"] = (
    pbp_shots["base_score"] *
    pbp_shots["home_mult"] *
    pbp_shots["playoff_mult"]
)

# =========================================
# 6. Aggregate per player (personId)
# =========================================
player_pressure = (
    pbp_shots.groupby("personId", as_index=False)
             .agg(raw_pressure=("pressure_score", "mean"))
             .rename(columns={"personId": "PLAYER_ID"})
)

# clean PLAYER_ID for merging
player_pressure["PLAYER_ID"] = pd.to_numeric(player_pressure["PLAYER_ID"], errors="coerce")
player_pressure = player_pressure.dropna(subset=["PLAYER_ID"])
player_pressure["PLAYER_ID"] = player_pressure["PLAYER_ID"].astype(int)

# =========================================
# 6.5: Introduce the marcos_2 variable
# =========================================
mean_raw = player_pressure["raw_pressure"].mean()
std_raw = player_pressure["raw_pressure"].std(ddof=0)

if std_raw == 0 or np.isnan(std_raw):
    # edge case: everyone same → all zeros
    player_pressure["marcos_2"] = 0.0
else:
    z = (player_pressure["raw_pressure"] - mean_raw) / std_raw
    scaled = z * 3.0  # ~±3σ → about ±9
    player_pressure["marcos_2"] = np.clip(scaled, -10, 10)


# =========================================
# 7. Merge into stats and collapse per player (one row)
# =========================================
stats_with_f6 = stats.merge(player_pressure[["PLAYER_ID", "marcos_2"]], on="PLAYER_ID", how="left")

final_feature6 = (
    stats_with_f6
    .groupby(["PLAYER_ID", "PLAYER_NAME"], as_index=False)
    .agg(marcos_2=("marcos_2", "mean"))
)

# =========================================
# 8. Save output
# =========================================
final_feature6.to_csv("stats_with_feature6_one_row_per_player.csv", index=False)

# Load your final Feature 6 dataset
df = pd.read_csv("stats_with_feature6_one_row_per_player.csv")

# Drop players with no clutch scoring events
df = df.dropna(subset=["marcos_2"])

# Sort descending by marcos_2
df_sorted = df.sort_values(by="marcos_2", ascending=False)

# Print top 20 players
print("Top 20 Players by Shot Clock Pressure Performance (Feature 6):\n")
print(df_sorted.head(20)[["PLAYER_ID", "PLAYER_NAME", "marcos_2"]])

f5 = pd.read_csv("stats_with_feature5_one_row_per_player.csv")
f6 = pd.read_csv("stats_with_feature6_one_row_per_player.csv")

# Merge on PLAYER_NAME (ignoring PLAYER_ID)
merged = f5.merge(f6, on="PLAYER_NAME", how="inner", suffixes=("_f5", "_f6"))

# Keep only relevant columns
final_df = merged[["PLAYER_NAME", "marcos_1", "marcos_2"]]

# Set index to PLAYER_NAME
final_df = final_df.set_index("PLAYER_NAME")

# =========================================
# 9. Standardize marcos_1 and marcos_2 to [-10, 10]
# =========================================

mm_scaler = MinMaxScaler(feature_range=(-10, 10))
mm_scaler.set_output(transform='pandas')
scaled_marcos_df = mm_scaler.fit_transform(X=final_df,y=None)

# Print the first few rows
print(scaled_marcos_df.head())

# Optionally save
scaled_marcos_df.to_csv("datasets/engineered-datasets/combined_marcos_features.csv")
print("Saved combined_marcos_features.csv")