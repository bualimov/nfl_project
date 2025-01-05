import pandas as pd

# Read the tracking data for Week 1
print("Reading tracking data...")
tracking_df = pd.read_csv('tracking_week_1.csv')
plays_df = pd.read_csv('plays.csv')

# Get play details
play_id = 64
game_id = tracking_df[tracking_df['playId'] == play_id]['gameId'].iloc[0]
play_details = plays_df[(plays_df['gameId'] == game_id) & (plays_df['playId'] == play_id)].iloc[0]

# Get unique teams in the play
teams = tracking_df[tracking_df['playId'] == play_id]['club'].unique()

print("\nPlay Details:")
print(f"Week: 1")
print(f"Teams: {teams[0]} vs {teams[1]}")
print("\nPlay Information:")
print(play_details)

# Get field position details from first frame
first_frame = tracking_df[
    (tracking_df['playId'] == play_id) & 
    (tracking_df['frameId'] == tracking_df[tracking_df['playId'] == play_id]['frameId'].min())
]
print("\nField Position Details:")
print(first_frame[['playDirection', 'x', 'y']].head()) 