import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from nba_api.stats.endpoints import LeagueDashPlayerClutch, PlayByPlayV2, playerdashboardbyclutch, leaguedashteamclutch
from nba_api.stats.static import players

"""
Two parts to this file:
1. Ridge regression model based on nine clutch features (in comparison to WPA)
to create a meta-feature for a player's "game-changing" ability in bad situations.

*THE OUTPUT VARIABLE:
- How well they affect the winning probability

*The clutch features:
- Clutch percentage of points responsible for (if this exists: vs. percentage of points responsible for giving up?)
- Clutch assist to turnover ratio
- Clutch Usage Rate (but NOT blowout usage rate)
- Shot taking in the clutch (not really heavily since we want to focus more on the results rather than pure confidence)
*contextual, non-statistical data:*
- Opponent strength (defensive and offensive rating of the opponent, with everything amplified in the playoffs)
- Away vs home game
- Shot clock pressure
- Whats on the line (playoffs wise → farther-in games in a series worth more)

*The time periods we are going over:
- Last 5 minutes of 4th quarter
- Any overtime period when the score differential is five points or less
- Blowout recovery --> when the score differential in a game is brought from 20 points or more to 5 points or less


2. An interactive visualization based on the "game-changer" output
"""

# --- 1. Get Player IDs and Clutch Data (Base DataFrame) ---
# NOTE: This only captures the official NBA clutch window (Last 5 min, score <= 5)
# Use a specific season ID, e.g., '22023' for the 2023-24 season

#get for all years you care about:
clutch_stats = LeagueDashPlayerClutch(
    clutch_time_period='Last 5 Minutes', 
    point_diff_nullable='5',
    season='2023-24' 
)
df_clutch = clutch_stats.get_data_frames()[0]

# Filter down to the essential stats for our X features
df_clutch = df_clutch[['PLAYER_ID', 'PLAYER_NAME', 'MIN', 'AST', 'TOV', 'USG_PCT', 'PTS', 'FGA', 'FG_PCT']]

# --- 2. MOCK COMPLEX & CONTEXTUAL DATA ---
# These variables (especially TOTAL_WPA and Blowout Recovery) MUST be calculated 
# by processing PBP data from endpoints like PlayByPlayV2
# and applying WPA models (which you must build separately).

N = len(df_clutch)
np.random.seed(42)

# Y (Target Variable for the Inner Ridge Model)
# TOTAL_WPA is the total change in Win Probability during ALL high-leverage periods.
# This would be calculated by summing WPA for each player's action.
df_clutch['TOTAL_WPA'] = np.random.normal(loc=0.5, scale=5.0, size=N) 

# X features that require external/complex calculation
df_clutch['RECOVERY_WPA_SCORE'] = np.random.uniform(0, 3, N)    # Blowout Recovery Impact
df_clutch['OPP_DEF_FACTOR'] = np.random.uniform(0.8, 1.2, N)     # X5: Opponent Strength Factor
df_clutch['PLAYOFF_LEVERAGE'] = np.random.uniform(1.0, 4.0, N)   # X6: Playoff Multiplier (1.0 to 4.0 for Finals)
df_clutch['SHOT_CLOCK_FACTOR'] = np.random.uniform(0.1, 0.5, N)  # X7: Time Pressure Factor (Higher value for more late-clock actions)
df_clutch['CLUTCH_FGA_VOLUME'] = df_clutch['FGA'] / df_clutch['MIN'].replace(0, 1)

# Ensure no NaNs before calculating X features
df_clutch = df_clutch.fillna(0)


# X1: Clutch Percentage of Points Responsible For
# Simple approximation: PTS + (2 * AST) as a percentage of overall team points when player is on floor
df_clutch['PTS_RESPONSIBLE_PCT'] = (df_clutch['PTS'] + (2 * df_clutch['AST'])) / df_clutch['MIN'].replace(0, 1)

# X2: Clutch Assist to Turnover Ratio
# Handle zero division: replace 0 turnovers with a tiny number for stability.
df_clutch['CLUTCH_A_TO'] = df_clutch['AST'] / df_clutch['TOV'].replace(0, 1e-6)

# X3: Clutch Usage Rate (USG_PCT is already provided as a raw number in the data)
df_clutch.rename(columns={'USG_PCT': 'CLUTCH_USAGE_RATE'}, inplace=True)



clutch_player_dashboard = playerdashboardbyclutch.PlayerDashboardByClutch(player_id='203999')
print('player dashboard made')
last_5_min = clutch_player_dashboard.expected_data['Last5MinPlusMinus5PointPlayerDashboard']
for index in last_5_min:
    print(f"{index}: ...")


# Define Features (X) and Target (Y)
X_cols = [
    'PTS_RESPONSIBLE_PCT', 'CLUTCH_A_TO', 'CLUTCH_USAGE_RATE', 
    'CLUTCH_FGA_VOLUME', 'RECOVERY_WPA_SCORE', 'OPP_DEF_FACTOR', 
    'PLAYOFF_LEVERAGE', 'SHOT_CLOCK_FACTOR'
]
X = df_clutch[X_cols]
Y = df_clutch['TOTAL_WPA']

# 1. Scale Features (Essential for Regularization methods like Ridge) 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Implement Ridge Regression
# We use a non-zero alpha (e.g., 1.0) to penalize large coefficients, 
# preventing one feature (like Clutch A/TO) from dominating due to high variance.
ridge_model = Ridge(alpha=1.0) 
ridge_model.fit(X_scaled, Y)

# 3. Calculate the Final CGI Score (The X3 Feature)
# The CGI score is the predicted WPA value from the Ridge model.
df_clutch['GAME_CHANGER_INDEX_CGI'] = ridge_model.predict(X_scaled)

# Display the Final Output and Learned Weights
print("--- Learned Weights (Betas) for the CGI ---")
weights_df = pd.DataFrame({
    'Feature': X_cols,
    'Learned_Weight': ridge_model.coef_
}).sort_values(by='Learned_Weight', ascending=False)
print(weights_df)

print("\n--- Final X3 Feature (Clutch Game-Changer Index - CGI) ---")
print(df_clutch[['PLAYER_NAME', 'GAME_CHANGER_INDEX_CGI']].sort_values(
    by='GAME_CHANGER_INDEX_CGI', ascending=False).head(5))