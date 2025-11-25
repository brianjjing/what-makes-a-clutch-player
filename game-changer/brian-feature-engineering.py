import pandas as pd, numpy as np
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

id_to_player_mapping = player_stats[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates().set_index('PLAYER_ID')['PLAYER_NAME'].to_dict()
print(id_to_player_mapping)

# Combine regular season and playoffs play-by-play data with is_playoffs flag
reg_season_pbp['is_playoffs'] = 'N'
playoffs_pbp['is_playoffs'] = 'Y'
all_pbp = pd.concat([reg_season_pbp, playoffs_pbp], ignore_index=True)
all_pbp = all_pbp.sort_values(by='personId')

print(player_stats.columns)

print(all_pbp.columns)


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
    scoring = scoring[scoring['points_on_play'] > 0]
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
print(ast_to_tov.head())


#FEATURE 3: CLUTCH USAGE RATE


#DATAFRAME CREATION:
# Create brian_df with all features
unique_players = player_stats['PLAYER_NAME'].unique()
brian_df = pd.DataFrame(index=unique_players)
# brian_1: Percentage of clutch points responsible for (with playoff multiplier)
brian_df['brian_1'] = points_responsible_pct.reindex(unique_players).fillna(0)

# brian_2: Clutch assist to turnover ratio
brian_df['brian_2'] = ast_to_tov.reindex(unique_players)

# brian_3: Clutch usage rate
#brian_df['brian_3'] = np.nan


#STANDARDIZATION: Standardizing all the features to the range (-10.0, 10.0)
mm_scaler = MinMaxScaler(feature_range=(-10, 10))
mm_scaler.set_output(transform='pandas')
scaled_brian_df = mm_scaler.fit_transform(X=brian_df,y=None)

print("\nBrian Scaled Features DataFrame:")
print(scaled_brian_df)





# #Function to get the total duration a team was playing in the clutch:
# def get_team_clutch_duration(group):
#     start = group['mins_left'].max()
#     end = group['mins_left'].min()
#     return start - end

# clutch_mins_by_team_df = all_pbp[['teamId', 'gameId', 'mins_left']] #should weigh playoffs more???

# #Applying the above function to get the playing duration for the teams in every game:
# clutch_period_durations = clutch_mins_by_team_df.groupby(['teamId', 'gameId']).apply(get_team_clutch_duration)
# print(f"Type of game_durations: {clutch_period_durations}")

# #Applying it to the whole team over the whole season:
# total_clutch_mins_by_team = clutch_period_durations.reset_index().groupby('teamId').sum().to_dict()


# total_team_clutch_mins = player_stats['TEAM_ID'].map(total_clutch_mins_by_team)

# # Calculate the Percentage
# # Formula: (Player's Minutes / Team's Total Available Clutch Minutes) * 100
# clutch_usage_rate = (
#     player_stats['MIN'] / total_team_clutch_mins
# ) * 100

# print(clutch_usage_rate)

# #print(player_stats[cols_to_show].sort_values(by='CLUTCH_PARTICIPATION_PCT', ascending=False).head())