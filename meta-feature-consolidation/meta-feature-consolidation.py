import pandas as pd

neil = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/shot-chart-mapping/data/neil_feature.csv')
print(neil.isna().sum())

varun = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/weighing-stats/data/meta-features/weighed_stats_feature.csv')
print(varun.isna().sum())

brianmarcos = pd.read_csv('/Users/brian/Documents/Python/what-makes-a-clutch-player/game-changer/datasets/meta-features/game_changer_var.csv')
print(brianmarcos.isna().sum())


print(neil.head(20))
print(varun.head(20))
print(brianmarcos.head(20))