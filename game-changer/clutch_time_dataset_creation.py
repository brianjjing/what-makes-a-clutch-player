import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from nba_api.stats.endpoints import LeagueGameFinder, LeagueDashPlayerClutch, PlayByPlayV3
from requests.exceptions import ReadTimeout
import regex as re
import time

"""
Game plan:
1. Ridge regression model based on nine clutch features (in comparison to WPA)
to create a meta-feature for a player's "game-changing" ability in bad situations.

*THE OUTPUT VARIABLE:
- How well they affect the winning probability

*The clutch features:


df = pd.DataFrame([]) <== all the data we need (im going to combine play by play data w overall player stats data)

Brian:
- ['brian_1']: Clutch percentage of points responsible for (if this exists: vs. percentage of points responsible for giving up?)
- ['brian_1']: Clutch assist to turnover ratio
- ['brian_1']: Clutch Usage Rate (but NOT blowout usage rate)
- ['brian_1']: Shot taking in the clutch (not really heavily since we want to focus more on the results rather than pure confidence)
*contextual, non-statistical data:*


Marcos:
- ['marcos_1']: Opponent strength (defensive and offensive rating of the opponent, with everything amplified in the playoffs)
- ['marcos_2']: Away vs home game
- ['marcos_3']: Shot clock pressure
- ['marcos_4']: Whats on the line (playoffs wise → farther-in games in a series worth more)




*The time periods we are going over:
- Last 5 minutes of 4th quarter
- Any overtime period when the score differential is five points or less
(NOT ANYMORE, IT'S UNFEASIBLE FOR OUR TIME FRAME): Blowout recovery --> when the score differential in a game is brought from 20 points or more to 5 points or less


2. An interactive visualization based on the "game-changer" output
"""

# --- 1. Getting clutch_player_stats dataframe ---
# seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

# clutch_player_stats = pd.DataFrame([])
# current_season_rows = [0,0]
# for season in seasons:
#     print(f"Season: {season}")
#     clutch_stats = LeagueDashPlayerClutch(
#         clutch_time='Last 5 Minutes', 
#         point_diff='10',
#         season=season
#     )
#     df_clutch = clutch_stats.get_data_frames()
#     current_season_df = df_clutch[0]
#     current_season_df['season'] = season
    
#     clutch_player_stats = pd.concat([clutch_player_stats, current_season_df])
#     print("Concatenated to df!")

# clutch_player_stats.to_csv(f'clutch_player_stats_since_2020_21.csv', index=False)
# print("clutch_player_stats df saved as .csv file!")
    

# Index(['GROUP_SET', 'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID',
#    'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM',
#    'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
#    'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
#    'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS', 'DD2', 'TD3',
#    'WNBA_FANTASY_PTS', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK',
#    'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK',
#    'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK',
#    'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK', 'TOV_RANK',
#    'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'PTS_RANK',
#    'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK', 'TD3_RANK',
#    'WNBA_FANTASY_PTS_RANK', 'TEAM_COUNT'],
#   dtype='object')


# # --- 2. Getting clutch_pbp_stats dataframe from the Kaggle dataframe ---

season_list = ['2020', '2021', '2022', '2023', '2024']

reg_season_clutch_pbp_df = pd.DataFrame([])
playoff_clutch_pbp_df = pd.DataFrame([])
for season_start_year in season_list:
    print(f"Season start year: {season_start_year}")
    
    #Reg season df:
    reg_season_df = pd.read_csv(f"game-changer/datasets/playbyplay/cdnnba_{season_start_year}.csv")
    reg_season_df['season_start_year'] = season_start_year
    reg_season_df['mins_left'] = reg_season_df['clock'].astype(str).str[2:4].astype(int)
    
    is_fourth = reg_season_df['period'] >= 4
    under_ten_point_diff = abs(reg_season_df['scoreHome'] - reg_season_df['scoreAway']) <= 10
    final_five_mins = reg_season_df['mins_left'] < 5
    
    reg_season_df = reg_season_df[is_fourth & under_ten_point_diff & final_five_mins]
    print("Regular season df made")
    
    
    #Playoff season df:
    playoff_season_df = pd.read_csv(f"game-changer/datasets/playbyplay/cdnnba_po_{season_start_year}.csv")
    playoff_season_df['season_start_year'] = season_start_year
    playoff_season_df['mins_left'] = playoff_season_df['clock'].astype(str).str[2:4].astype(int)
    
    is_fourth = playoff_season_df['period'] >= 4
    under_ten_point_diff = abs(playoff_season_df['scoreHome'] - playoff_season_df['scoreAway']) <= 10
    final_five_mins = playoff_season_df['mins_left'] < 5
    
    playoff_season_df = playoff_season_df[is_fourth & under_ten_point_diff & final_five_mins]
    print("Playoff df made")
    
    
    #Concat the dfs
    reg_season_clutch_pbp_df = pd.concat([reg_season_clutch_pbp_df, reg_season_df])
    print("Regular Season clutch pbp added to df!")
    playoff_clutch_pbp_df = pd.concat([playoff_clutch_pbp_df, playoff_season_df])
    print("Playoff clutch pbp added to df!")
    
reg_season_clutch_pbp_df.to_csv('game-changer/datasets/reg_season_pbp_since_2020_21.csv', index=False)
print("reg_season_clutch_pbp_df saved as .csv file!")
playoff_clutch_pbp_df.to_csv('game-changer/datasets/playoffs_pbp_since_2020_21.csv', index=False)
print("playoff_clutch_pbp_df saved as .csv file!")



"""
Below is the shitty scraping algorithm I tried to use at first. Just keeping it for future reference.
"""

# #HELPER FUNCTIONS:
# #Getting game ids:
# def get_game_ids(season):
#     games = LeagueGameFinder(season_nullable=season).get_data_frames()[0]
#     return games
# #Getting score differential:
# def parse_score_diff_abs(scoreHome, scoreAway):
#     # Check for NaN values before calculating
#     if pd.isna(scoreHome) or pd.isna(scoreAway) or scoreHome=='' or scoreAway=='':
#         return np.nan
#     return abs(int(scoreHome) - int(scoreAway))
# #Getting minutes left in quarter:
# def parse_time_minutes_left(duration_str):
#     if pd.isna(duration_str):
#         return np.nan
#     # Use regex to find minutes (M) and seconds (S) for robust extraction
#     match = re.search(r'PT(\d+M)?(\d+\.?\d*S)?', duration_str)
#     minutes = int(match.group(1)[:-1]) if match.group(1) else 0
#     seconds = float(match.group(2)[:-1]) if match.group(2) else 0
    
#     return minutes + (seconds / 60)


# #specify which game id was skipped too!


# #Seasons loop:
# for season in seasons_v2:
#     print(f'Season {season}')
    
#     game_ids = get_game_ids(season)
#     clutch_pbps = pd.DataFrame([])
#     total_games = len(game_ids)
#     current_game_num = 1
#     consecutive_failure_count = 0
#     num_skipped = 0
#     game_ids_skipped = []
    
#     for game_id in game_ids['GAME_ID']:
#         print(f"Game {current_game_num} / {total_games}:")
#         print(f"Game ID: {game_id}")
#         pbp = pd.DataFrame()
        
#         try:
#             # 1. Set the 30-second timeout for the request
#             pbp = PlayByPlayV3(
#                 game_id=game_id, 
#                 start_period=4, 
#                 end_period=10, 
#                 timeout=CUSTOM_TIMEOUT # The explicit 30-second timeout
#             ).get_data_frames()[0]
            
#         except ReadTimeout as e:
#             # 2. Catch the timeout error and skip the game
#             consecutive_failure_count += 1
        
#             if consecutive_failure_count >= MAX_CONSECUTIVE_FAILURES:
#                 print(f"\n🚨 SUSTAINED BLOCK DETECTED! Pausing for {SUSTAINED_BLOCK_COOLDOWN} seconds to reset API throttle.")
#                 time.sleep(SUSTAINED_BLOCK_COOLDOWN)
            
#             print(f"Skipping Game {game_id}: ReadTimeout occurred after {CUSTOM_TIMEOUT}s. Moving to next game.")
#             # Add a longer pause after a failure before the next game starts, to avoid API throttling
#             time.sleep(POST_FAILURE_COOLDOWN) 
#             current_game_num += 1
#             num_skipped += 1
#             game_ids_skipped.append(game_id)
#             continue # <-- Skips processing and jumps to the next game_id
            
#         except Exception as e:
#             # Handle other fatal errors (e.g., connection errors)
#             print(f"Skipping Game {game_id}: Fatal Error ({e}). Moving to next game.")
#             current_game_num += 1
#             consecutive_failure_count = 0 # Assume this failure is unique
#             num_skipped += 1
#             game_ids_skipped.append(game_id)
#             continue
        
#         if not pbp.empty:
#             consecutive_failure_count = 0
#             pbp['point_differential'] = pbp.apply(
#                 lambda row: parse_score_diff_abs(row['scoreHome'], row['scoreAway']), 
#                 axis=1
#             )
#             pbp['mins_remaining'] = pbp['clock'].apply(parse_time_minutes_left)

#             clutch_pbp = pbp[(pbp['point_differential']<=10) & (pbp['mins_remaining']<=5)]
#             print(clutch_pbp)
#             clutch_pbps = pd.concat([clutch_pbps, clutch_pbp])
            
#             current_game_num += 1
#             time.sleep(RATE_LIMIT_PAUSE) # Maintain the rate limit pause
    
#     season_renamed = season.replace('-', '_')
#     try:
#         with open(f"game-changer/datasets/num_skipped_{season_renamed}.txt", "w") as f:
#             f.write(f"{num_skipped}\n")
#             f.write(",".join(game_ids_skipped))
#     except:
#         with open(f"num_skipped_{season_renamed}.txt", "w") as f:
#             f.write(f"{num_skipped}\n")
#             f.write(",".join(game_ids_skipped))
#     try:
#         clutch_pbps.to_csv(f'game-changer/datasets/clutch_pbp_data_{season_renamed}.csv', index=False)
#     except:
#         clutch_pbps.to_csv(f'clutch_pbp_data_{season_renamed}.csv', index=False)

#     # Index(['gameId', 'actionNumber', 'clock', 'period', 'teamId', 'teamTricode',
#     #        'personId', 'playerName', 'playerNameI', 'xLegacy', 'yLegacy',
#     #        'shotDistance', 'shotResult', 'isFieldGoal', 'scoreHome', 'scoreAway',
#     #        'pointsTotal', 'location', 'description', 'actionType', 'subType',
#     #        'videoAvailable', 'shotValue', 'actionId'],
#     #       dtype='object')