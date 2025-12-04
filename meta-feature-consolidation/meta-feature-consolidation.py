import pandas as pd
from sklearn.decomposition import PCA

neil = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/shot-chart-mapping/data/neil_feature.csv')
neil = neil[['player', 'eFG_STD']]
neil.columns = ['PLAYER_NAME', 'eFG_STD']
print(neil.isna().sum())

varun = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/weighing-stats/data/meta-features/weighed_stats_feature.csv')
print(varun.isna().sum())

brianmarcos = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/game-changer/datasets/meta-features/game_changer_var.csv')
brianmarcos.columns = ['PLAYER_NAME', 'GAME_CHANGER_STD']
print(brianmarcos.isna().sum())

print(neil.head(20))
print(varun.head(20))
print(brianmarcos.head(20))

meta_features = neil
meta_features = meta_features.merge(varun, on='PLAYER_NAME', how='inner')
meta_features = meta_features.merge(brianmarcos, on='PLAYER_NAME', how='inner')
meta_features.index = meta_features['PLAYER_NAME']
meta_features = meta_features.drop('PLAYER_NAME', axis=1)

print(meta_features)

pca = PCA(n_components=1)

# 3. Fit and Transform
# This creates your single "Clutch Variable"
clutch_component = pca.fit_transform(meta_features[['eFG_STD', 'CLUTCH_SCORE_STD', 'GAME_CHANGER_STD']])

# 4. Check the direction!
# PCA doesn't know "good" from "bad". It might make high efficiency negative.
# Look at the "loadings" (weights) given to your features.
loadings = pca.components_[0]
print("Feature Weights:", dict(zip(meta_features.columns, loadings)))

# If the weight for a known positive feature (like Efficiency) is negative,
# flip the sign of your result.
if loadings[1] < 0: # Assuming index 1 is Efficiency
    clutch_component = -clutch_component

# 5. Scale to your desired range (e.g., 0 to 100 or -10 to 10)
# final_scaler = MinMaxScaler(feature_range=(0, 100))
# final_clutch_score = final_scaler.fit_transform(clutch_component)

# Add to your dataframe
meta_features['FINAL_CLUTCH_SCORE'] = clutch_component
meta_features = meta_features.sort_values(by='FINAL_CLUTCH_SCORE', ascending=False)
print(meta_features.head(25))