from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams
import pandas as pd


# - Clutch percentage of points responsible for (if this exists: vs. percentage of points responsible for giving up?)
# - Clutch assist to turnover ratio
# - Clutch Usage Rate (but NOT blowout usage rate)
# - Shot taking in the clutch (not really heavily since we want to focus more on the results rather than pure confidence)

player_stats = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/game-changer/datasets/clutch_player_stats_since_2020_21.csv')
playoffs_pbp = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/game-changer/datasets/playoffs_pbp_since_2020_21.csv')
reg_season_pbp = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/game-changer/datasets/reg_season_pbp_since_2020_21.csv')

brian_df = pd.DataFrame([])
brian_df['brian_1'] = ...