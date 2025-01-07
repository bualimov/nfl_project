import pandas as pd

# Read the tracking data for Week 1
print("Reading tracking data...")
tracking_df = pd.read_csv('tracking_week_1.csv')

# Print column names
print("\nColumn names:")
for col in tracking_df.columns:
    print(col)

# Print sample data for play 64
print("\nSample data for play 64:")
play_df = tracking_df[tracking_df['playId'] == 64].head()
print(play_df) 