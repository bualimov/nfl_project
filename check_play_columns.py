import pandas as pd

# Read the plays data
print("Reading plays data...")
plays_df = pd.read_csv('plays.csv')

# Print column names
print("\nColumn names in plays.csv:")
for col in plays_df.columns:
    print(col)

# Print a sample row
print("\nSample row:")
print(plays_df.iloc[0]) 