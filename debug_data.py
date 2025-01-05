import pandas as pd
import numpy as np

# Read data
print("Reading data...")
tracking_df = pd.read_csv('tracking_week_1.csv')
plays_df = pd.read_csv('plays.csv')
players_df = pd.read_csv('players.csv')

# Print column information
print("\nTracking data columns:", tracking_df.columns.tolist())
print("\nPlayers data columns:", players_df.columns.tolist())

# Get SEA vs DEN game
sea_den_plays = plays_df[
    ((plays_df['possessionTeam'] == 'SEA') & (plays_df['defensiveTeam'] == 'DEN')) |
    ((plays_df['possessionTeam'] == 'DEN') & (plays_df['defensiveTeam'] == 'SEA'))
]
sea_den_game = sea_den_plays['gameId'].iloc[0]

# Filter for play 64
play_df = tracking_df[
    (tracking_df['gameId'] == sea_den_game) & 
    (tracking_df['playId'] == 64)
].copy()

print("\nSample of tracking data for play 64:")
print(play_df[['nflId', 'club', 'jerseyNumber', 'displayName']].head())

# Rename tracking data displayName
play_df = play_df.rename(columns={'displayName': 'displayName_tracking'})

# Try merge
print("\nAttempting merge...")
merged_df = play_df.merge(
    players_df[['nflId', 'position', 'displayName']], 
    on='nflId', 
    how='left'
)

print("\nMerged data columns:", merged_df.columns.tolist())
print("\nSample of merged data:")
print(merged_df[['nflId', 'club', 'jerseyNumber', 'position', 'displayName_tracking', 'displayName']].head()) 