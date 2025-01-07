import pandas as pd
import numpy as np

def calculate_player_entropy(player_data, all_defenders):
    # Constants for entropy calculation
    w_theta = 0.3  # Weight for orientation
    w_v = 0.2      # Weight for velocity
    v_max = 10.0   # Maximum velocity for normalization (yards/second)
    grid_size = 1.0  # 1-yard grid
    sigma = 2.0    # Gaussian spread parameter
    
    # Get player position and movement data
    x = player_data['x']
    y = player_data['y']
    theta = player_data['dir']  # Player's facing direction
    v = np.sqrt(player_data['s']**2)  # Using speed only, excluding acceleration
    
    # Calculate angle relative to ball (assuming ball is at center of field width)
    ball_y = 26.65  # Center of field width (53.3/2)
    theta_relative = np.abs(theta - np.degrees(np.arctan2(ball_y - y, 60 - x)))
    
    # Create grid for field discretization
    x_grid = np.arange(max(0, x-10), min(120, x+10), grid_size)
    y_grid = np.arange(max(0, y-10), min(53.3, y+10), grid_size)
    
    # Calculate base position probability
    p = np.zeros((len(x_grid), len(y_grid)))
    for def_x, def_y in all_defenders[['x', 'y']].values:
        for i, gx in enumerate(x_grid):
            for j, gy in enumerate(y_grid):
                dist = np.sqrt((gx-def_x)**2 + (gy-def_y)**2)
                p[i,j] += np.exp(-dist**2/(2*sigma**2))
    
    # Normalize probabilities
    p = p / np.sum(p)
    
    # Calculate orientation factor
    orientation_factor = 1 + w_theta * np.cos(np.radians(theta_relative))
    
    # Calculate velocity factor
    velocity_factor = 1 + w_v * (min(v, v_max) / v_max)
    
    # Calculate entropy with modifiers
    base_entropy = -np.sum(p * np.log2(p + 1e-10))
    total_entropy = base_entropy * orientation_factor * velocity_factor
    
    return total_entropy

# Read data
print("Reading data...")
tracking_df = pd.read_csv('tracking_week_1.csv')
plays_df = pd.read_csv('plays.csv')
players_df = pd.read_csv('players.csv')

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

# Get play details
play_details = plays_df[
    (plays_df['gameId'] == sea_den_game) & 
    (plays_df['playId'] == 64)
].iloc[0]

# Merge with players data
play_df = play_df.merge(
    players_df[['nflId', 'position', 'displayName']], 
    on='nflId', 
    how='left'
)

# Get defensive team
defensive_team = play_details['defensiveTeam']

# Find snap frames
snap_frame = play_df[play_df['event'] == 'ball_snap']['frameId'].min()
line_set_frame = play_df[play_df['event'] == 'line_set']['frameId'].min()

# Filter for pre-snap frames
play_df = play_df[
    (play_df['frameId'] >= line_set_frame) & 
    (play_df['frameId'] <= snap_frame)
].copy()

# Initialize results dictionary
entropy_results = {}

# Calculate entropy for each frame
for frame in sorted(play_df['frameId'].unique()):
    frame_data = play_df[play_df['frameId'] == frame].copy()
    defense_data = frame_data[frame_data['club'] == defensive_team].copy()
    
    # Calculate entropy for each defensive player
    for _, player in defense_data.iterrows():
        player_id = player['nflId']
        if player_id not in entropy_results:
            entropy_results[player_id] = {
                'position': player['position'],
                'number': int(player['jerseyNumber']) if not pd.isna(player['jerseyNumber']) else 0,
                'name': player['displayName'],
                'entropy_values': []
            }
        
        entropy = calculate_player_entropy(player, defense_data)
        entropy_results[player_id]['entropy_values'].append(entropy)

# Create DataFrame for results
results = []
for player_id, data in entropy_results.items():
    player_row = {
        'Position': data['position'],
        'Number': data['number'],
        'Player': data['name']
    }
    # Add entropy values for each frame
    for i, entropy in enumerate(data['entropy_values']):
        frame_num = line_set_frame + i
        player_row[f'Frame_{frame_num}'] = entropy
    results.append(player_row)

results_df = pd.DataFrame(results)

# Calculate average entropy for each player
results_df['Avg_Entropy'] = results_df.filter(like='Frame_').mean(axis=1)

# Sort by position and number
results_df = results_df.sort_values(['Position', 'Number'])

# Format the DataFrame
print("\nEntropy values for defensive players (Play 64, SEA vs DEN):")
print(f"Frames from {line_set_frame} (line set) to {snap_frame} (ball snap)")
print("\nPlayer Information and Average Entropy:")
print(results_df[['Position', 'Number', 'Player', 'Avg_Entropy']].to_string(index=False))

print("\nDetailed entropy values by frame:")
# Display frame-by-frame values with better formatting
frame_columns = [col for col in results_df.columns if col.startswith('Frame_')]
for _, row in results_df.iterrows():
    print(f"\n{row['Position']} #{row['Number']} - {row['Player']}:")
    for frame in frame_columns:
        frame_num = int(frame.split('_')[1])
        print(f"Frame {frame_num}: {row[frame]:.3f}") 