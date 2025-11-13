from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams
import pandas as pd

all_teams = teams.get_teams()
print(all_teams)

player_dictionary = [player for player in players.get_players() if player["is_active"]]
print(player_dictionary)

year_2025 = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/shot-chart-mapping/years/2025.csv')
print(year_2025.columns)

who_made_most = year_2025[['PLAYER_NAME', 'SHOT_MADE_FLAG']].groupby('PLAYER_NAME').sum()
who_made_most = who_made_most.sort_values(by='SHOT_MADE_FLAG', ascending=False)
print(who_made_most)

print(len(year_2025['TEAM_NAME'].value_counts()))