import pandas as pd

# Read plays data
print("Reading plays data...")
plays_df = pd.read_csv('plays.csv')

# Print column names
print("\nColumns in plays.csv:")
for col in plays_df.columns:
    print(f"- {col}")

# Print sample data for SEA vs DEN game
print("\nSample play data:")
print(plays_df.iloc[0]) 