import pandas as pd

# Read the tracking data
print("Reading data...")
tracking_df = pd.read_csv('tracking_week_1.csv')

# Filter for play 64
play_df = tracking_df[tracking_df['playId'] == 64].copy()

# Look at events
print("\nUnique events in play:")
print(play_df['event'].unique())

# Find snap frame
snap_frame = play_df[play_df['event'] == 'ball_snap']['frameId'].iloc[0]
print(f"\nSnap frame: {snap_frame}")

# Show some frames before and after snap
print("\nFrames around snap:")
for frame in range(snap_frame - 5, snap_frame + 5):
    events = play_df[play_df['frameId'] == frame]['event'].unique()
    print(f"Frame {frame}: {events}") 