import pandas as pd, numpy as np
from nba_api.stats.static import players
from sklearn.preprocessing import MinMaxScaler
"""
Game plan:
1. Ridge regression model based on nine clutch features (in comparison to WPA)
to create a meta-feature for a player's "game-changing" ability in bad situations.

*THE OUTPUT VARIABLE:
- How well they affect the winning probability

*The clutch features:
BRIAN:
- Clutch percentage of points responsible for (if this exists: vs. percentage of points responsible for giving up?)
- Clutch assist to turnover ratio
- Clutch Usage Rate (but NOT blowout usage rate)
- Shot taking in the clutch (not really heavily since we want to focus more on the results rather than pure confidence)
MARCOS:
- Opponent strength (defensive and offensive rating of the opponent)
- Pressure score

2. An interactive visualization based on the "game-changer" output
"""

player_stats = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/game-changer/datasets/clutch_player_stats_since_2020_21.csv')
player_stats = player_stats.sort_values(by='PLAYER_ID')
playoffs_pbp = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/game-changer/datasets/playoffs_pbp_since_2020_21.csv')
playoffs_pbp = playoffs_pbp.sort_values(by='personId')
reg_season_pbp = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/game-changer/datasets/reg_season_pbp_since_2020_21.csv')
reg_season_pbp = reg_season_pbp.sort_values(by='personId')

# Combine regular season and playoffs play-by-play data with is_playoffs flag
reg_season_pbp['is_playoffs'] = 'N'
playoffs_pbp['is_playoffs'] = 'Y'
all_pbp = pd.concat([reg_season_pbp, playoffs_pbp], ignore_index=True)
all_pbp = all_pbp.sort_values(by='personId')

#print(player_stats.columns)
print(all_pbp.columns)
print(all_pbp.head())


nba_player_list = players.get_players()
id_to_player_mapping = {p['id']: p['full_name'] for p in nba_player_list}

print(f"Mapped {len(id_to_player_mapping)} unique IDs to players from NBA API:")
print(id_to_player_mapping)


#FEATURE 1: PERC OF POINTS RESPONSIBLE FOR
def calculate_points_responsible_pct(pbp_df):
    """Calculate percentage of clutch points each player was responsible for (scored)."""
    pbp = pbp_df.sort_values(['gameId', 'actionNumber']).copy() #Index it by the game and the action sequence #
    pbp['pointsTotal'] = pbp['pointsTotal'].fillna(0)
    scoring = pbp[pbp['playerName'].notna()].copy()
    
    # Calculate points on each play (pointsTotal is cumulative)
    scoring['points_on_play'] = scoring.groupby(['gameId', 'personId'], dropna=False)['pointsTotal'].diff().fillna(
        scoring.groupby(['gameId', 'personId'], dropna=False)['pointsTotal'].transform('first') #Fills first row with itself
    )
    team_totals = scoring.groupby('teamId')['points_on_play'].sum() #for division for proportions
    
    # Calculating the total that each player was responsible for, as a prop of their team's:
    player_team_scoring_percentage = {}
    for person_id, group in scoring[scoring['personId'].notna()].groupby('personId'):
        player_points_sum = group['points_on_play'].sum()
        team_id = group['teamId'].iloc[0]
        team_total_points = team_totals.get(team_id)
        name = id_to_player_mapping.get(person_id)
        player_team_scoring_percentage[name] = player_points_sum / team_total_points
    
    
    #Calculating clutch usage rates per player:
    for person_id, group in scoring[scoring['personId'].notna()].groupby('personId'):
        player_points_sum = group['points_on_play'].sum()
        team_id = group['teamId'].iloc[0]
        team_total_points = team_totals.get(team_id)
        name = id_to_player_mapping.get(person_id)
        player_team_scoring_percentage[name] = player_points_sum / team_total_points
    
    return pd.Series(player_team_scoring_percentage)

points_responsible_pct = calculate_points_responsible_pct(all_pbp)
print("\nPercentage of Clutch Points Responsible For:")
print(points_responsible_pct.sort_values(ascending=False).head(20))


#FEATURE 2: AST-TOV RATIO:
player_stats_grouped = player_stats[['PLAYER_ID', 'AST', 'TOV']].groupby('PLAYER_ID').sum()
ast_to_tov = player_stats_grouped['AST'] / (player_stats_grouped['TOV'] + 1)
ast_to_tov = ast_to_tov.rename(index=id_to_player_mapping)
print("\nAssist-to-Turnover Ratio:")
print(ast_to_tov.sort_values(ascending=False).head(20))


#FEATURE 3: CLUTCH USAGE RATE
def calculate_clutch_usage_rate(pbp_df):
    player_possessions_by_game = pbp_df.groupby(['personId', 'teamTricode', 'gameId']).count()[['actionNumber']].rename(columns={'actionNumber': 'total_player_possessions'}).reset_index().set_index(['teamTricode', 'gameId'])
    print(player_possessions_by_game.iloc[0:20])

    team_possessions_by_game = pbp_df.groupby(['teamTricode', 'gameId']).count()[['actionNumber']].rename(columns={'actionNumber': 'total_team_possessions'})
    print(team_possessions_by_game.iloc[0:20])

    clutch_usage_rate = player_possessions_by_game.merge(team_possessions_by_game, on=['teamTricode', 'gameId'], how='left').reset_index().set_index(['personId'])
    clutch_usage_rate = clutch_usage_rate.reset_index()[['personId', 'total_player_possessions', 'total_team_possessions']].groupby('personId').sum()
    clutch_usage_rate['clutch_usage_rate'] = clutch_usage_rate['total_player_possessions'] / clutch_usage_rate['total_team_possessions']
    clutch_usage_rate = clutch_usage_rate['clutch_usage_rate'].rename(index=id_to_player_mapping)

    return clutch_usage_rate.sort_values(ascending=False)

clutch_usage_rate_series = calculate_clutch_usage_rate(all_pbp)
print("\nClutch Usage Rate By Player:")
print(clutch_usage_rate_series.head(20))

#DATAFRAME CREATION:
# Create brian_df with all features
# unique_players = player_stats['PLAYER_NAME'].unique()
# print(unique_players)

unique_player_names = player_stats['PLAYER_NAME'].unique()
brian_df = pd.DataFrame(index=unique_player_names)
# brian_1: Percentage of clutch points responsible for (with playoff multiplier)
brian_df['brian_1'] = points_responsible_pct.reindex(unique_player_names)

# brian_2: Clutch assist to turnover ratio
brian_df['brian_2'] = ast_to_tov.reindex(unique_player_names)

# brian_3: Clutch usage rate
brian_df['brian_3'] = clutch_usage_rate_series.reindex(unique_player_names)


#STANDARDIZATION: Standardizing all the features to the range (-10.0, 10.0)
mm_scaler = MinMaxScaler(feature_range=(-10, 10))
mm_scaler.set_output(transform='pandas')
scaled_brian_df = mm_scaler.fit_transform(X=brian_df,y=None).dropna()

print("\nBrian Scaled Features DataFrame:")
print(scaled_brian_df)