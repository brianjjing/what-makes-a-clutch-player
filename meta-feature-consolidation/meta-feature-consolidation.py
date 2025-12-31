import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Loading Data
try:
    neil = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/shot-chart-mapping/data/neil_feature.csv')
    varun = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/weighing-stats/data/meta-features/weighed_stats_feature.csv')
    brianmarcos = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/game-changer/datasets/meta-features/game_changer_var.csv')
except FileNotFoundError:
    # Creating dummy data for demonstration if files aren't found
    print("Warning: Files not found. Using dummy data for demonstration.")
    data = {'PLAYER_NAME': [f'Player {i}' for i in range(100)],
            'eFG_STD': [i * 0.1 for i in range(100)],
            'PLAYER_STATS_WEIGHED': [i * 0.2 for i in range(100)],
            'GAME_CHANGER_STD': [i * 0.15 for i in range(100)]}
    neil = pd.DataFrame({'player': data['PLAYER_NAME'], 'eFG_STD': data['eFG_STD']})
    varun = pd.DataFrame({'PLAYER_NAME': data['PLAYER_NAME'], 'weighed_stats': data['PLAYER_STATS_WEIGHED']})
    brianmarcos = pd.DataFrame({'PLAYER_NAME': data['PLAYER_NAME'], 'game_changer': data['GAME_CHANGER_STD']})

# Clean and Prepare Columns
neil = neil[['player', 'eFG_STD']]
neil.columns = ['PLAYER_NAME', 'eFG_STD']

varun.columns = ['PLAYER_NAME', 'PLAYER_STATS_WEIGHED']

brianmarcos.columns = ['PLAYER_NAME', 'GAME_CHANGER_STD']

# Merging datasets for features
meta_features = neil
meta_features = meta_features.merge(varun, on='PLAYER_NAME', how='inner')
meta_features = meta_features.merge(brianmarcos, on='PLAYER_NAME', how='inner')
meta_features.index = meta_features['PLAYER_NAME']
meta_features_clean = meta_features.drop('PLAYER_NAME', axis=1).dropna()

# Principal Component Analysis for valuation of clutch score:
pca = PCA(n_components=1)
clutch_component = pca.fit_transform(meta_features_clean[['eFG_STD', 'PLAYER_STATS_WEIGHED', 'GAME_CHANGER_STD']])

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Checking the direction of the PCA line (If high efficiency means a positive score)
loadings = pca.components_
print(loadings)
if sum(loadings[0]) < 0: 
# Can't force all axis values of the component line to be positive because of some potentially alternating signs, you can only flip the whole line. So only flip the line when it is MOSTLY negative.
    clutch_component = -clutch_component

# Add to DataFrame
meta_features_clean['FINAL_CLUTCH_SCORE'] = clutch_component

# Scale for readability (0 to 100)
scaler = MinMaxScaler(feature_range=(0, 100))
meta_features_clean['FINAL_CLUTCH_SCORE_SCALED'] = scaler.fit_transform(meta_features_clean[['FINAL_CLUTCH_SCORE']])

# Ridge Regression to display weights of each feature (importance of each / do any bring the score down? )
X_ridge = meta_features_clean[['eFG_STD', 'PLAYER_STATS_WEIGHED', 'GAME_CHANGER_STD']]
y_ridge = meta_features_clean['FINAL_CLUTCH_SCORE']

ridge = Ridge(alpha=1.0)
ridge.fit(X_ridge, y_ridge)

weights = ridge.coef_
intercept = ridge.intercept_

print("\n--- Ridge Regression Weights (Feature Importance) ---")
print(f"Intercept: {intercept:.4f}")
for feature, weight in zip(X_ridge.columns, weights):
    print(f"  {feature}: {weight:.4f}")

# Final Outputted Clutch Scires:
meta_features_clean = meta_features_clean.sort_values(by='FINAL_CLUTCH_SCORE', ascending=False)
print("\n--- Top 10 Clutch Players ---")
print(meta_features_clean[['FINAL_CLUTCH_SCORE_SCALED']].head(10))


#PCA captures the line w the most variance in the data. A player's score is the coordinate of that player on the specific line.