import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

player_stats = pd.read_csv('datasets/clutch_player_stats_since_2020_21.csv')
# playoffs_pbp = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/game-changer/datasets/playoffs_pbp_since_2020_21.csv')
# playoffs_pbp = playoffs_pbp.sort_values(by='personId')
# reg_season_pbp = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/game-changer/datasets/reg_season_pbp_since_2020_21.csv')
# reg_season_pbp = reg_season_pbp.sort_values(by='personId')
# all_pbp = pd.concat([reg_season_pbp, playoffs_pbp], ignore_index=True)
# all_pbp = all_pbp.sort_values(by='personId')
brian_features = pd.read_csv('datasets/engineered-datasets/combined_brian_features.csv')
marcos_features = pd.read_csv('datasets/engineered-datasets/combined_marcos_features.csv')
print(marcos_features.isna().sum())

game_changer = brian_features.merge(marcos_features, left_on='Unnamed: 0', right_on='PLAYER_NAME').drop('Unnamed: 0', axis=1).set_index('PLAYER_NAME')

#print(brian_features.head(20))
#print(marcos_features.head(20))
# nas = game_changer.isna()
# players = nas[nas['marcos_2']==True].index
# for player in players:
#     print(player)
print(game_changer)
game_changer_index = game_changer.index
# print(all_pbp.columns)

scaler = StandardScaler()
game_changer_scaled = pd.DataFrame(scaler.fit_transform(game_changer))
game_changer_scaled = game_changer_scaled.set_index(game_changer_index)
game_changer_scaled = game_changer_scaled.dropna()
game_changer_scaled.columns = ['brian_1', 'brian_2', 'brian_3', 'marcos_1', 'marcos_2']

print(game_changer_scaled)

game_changer_scaled_index = game_changer_scaled.index

pca = PCA(n_components = 1)
game_changer_var = pca.fit_transform(game_changer_scaled)
game_changer_var = pd.DataFrame(game_changer_var)
game_changer_var = game_changer_var.set_index(game_changer_scaled_index)
game_changer_var.columns = ['game_changer']

game_changer_var = pd.DataFrame(scaler.fit_transform(game_changer_var))

print(game_changer_var)
print(game_changer_var.max())
print(game_changer_var.min())

game_changer_var.to_csv('datasets/meta-features/game_changer_var.csv')
print('Set game_changer_var csv file!!!')