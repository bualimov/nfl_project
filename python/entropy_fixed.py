import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    
    # Calculate angle relative to ball
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

print("Reading data...")
tracking_df = pd.read_csv('tracking_week_1.csv')
plays_df = pd.read_csv('plays.csv')
players_df = pd.read_csv('players.csv')

# Get SEA vs DEN game from Week 1
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

# Calculate entropy for each player at each frame
entropy_data = {}
frames = sorted(play_df['frameId'].unique())

for frame in frames:
    frame_data = play_df[play_df['frameId'] == frame].copy()
    defense_data = frame_data[frame_data['club'] == defensive_team].copy()
    
    for _, player in defense_data.iterrows():
        player_id = f"{player['position']} #{int(player['jerseyNumber'])}"
        if player_id not in entropy_data:
            entropy_data[player_id] = {
                'position': player['position'],
                'entropy_values': [],
                'frames': []
            }
        
        entropy = calculate_player_entropy(player, defense_data)
        entropy_data[player_id]['entropy_values'].append(entropy)
        entropy_data[player_id]['frames'].append(frame)

# Filter for specific positions (CB #23 and DE #99)
selected_players = ['CB #23', 'DE #99']
filtered_entropy_data = {
    player_id: data 
    for player_id, data in entropy_data.items() 
    if player_id in selected_players
}

# Create figure
fig = plt.figure(figsize=(15, 8))

# Add play details at the top
play_text = (f"NFL Week 1, 2022: {play_details['possessionTeam']} vs {play_details['defensiveTeam']} - Play #{play_details['playId']}\n"
            f"Q{play_details['quarter']} {play_details['gameClock']} - {play_details['down']} & {play_details['yardsToGo']}")
plt.figtext(0.5, 0.98, play_text, ha='center', va='top', fontsize=12)

# Create main axis
ax = plt.subplot2grid((10, 1), (1, 0), rowspan=8)

# Set title and labels
plt.title('Defensive Player Entropy Over Time (Pre-snap Analysis)\nComparing CB #23 and DE #99', 
         fontsize=14, pad=15, weight='bold')
plt.xlabel('Frame Number', fontsize=12)
plt.ylabel('Entropy (bits)', fontsize=12)

# Set axis limits
plt.xlim(line_set_frame, snap_frame)
plt.ylim(5, 11)

# Add grid
plt.grid(True, alpha=0.3)

# Color scheme (two shades of red)
player_colors = {
    'CB #23': '#ff0000',  # Bright red
    'DE #99': '#8b0000'   # Dark red
}

# Initialize lines and labels
lines = {}
markers = {}
labels = {}
frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                    verticalalignment='top')

# Create dummy lines for legend
legend_lines = []
legend_labels = []
for player_id, color in player_colors.items():
    line = plt.Line2D([0], [0], color=color, linewidth=2, label=player_id)
    legend_lines.append(line)
    legend_labels.append(player_id)

# Add legend
ax.legend(handles=legend_lines, labels=legend_labels,
         bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

def init():
    for player_id, data in filtered_entropy_data.items():
        color = player_colors[player_id]
        
        # Create line without label (legend already created)
        line, = ax.plot([], [], color=color, linewidth=2)
        lines[player_id] = line
        
        # Create marker
        marker, = ax.plot([], [], 'o', color=color, markersize=8)
        markers[player_id] = marker
        
        # Create label
        label = ax.text(0, 0, '', color=color, fontsize=12,
                      bbox=dict(facecolor='white', edgecolor=color, alpha=0.7, pad=2))
        labels[player_id] = label
    
    return list(lines.values()) + list(markers.values()) + list(labels.values()) + [frame_text]

def update(frame_idx):
    frame = frames[frame_idx]
    frame_text.set_text(f'Frame: {frame}')
    
    for player_id, data in filtered_entropy_data.items():
        # Get data up to current frame
        current_frames = [f for f in data['frames'] if f <= frame]
        current_entropy = data['entropy_values'][:len(current_frames)]
        
        # Update line
        lines[player_id].set_data(current_frames, current_entropy)
        
        # Update marker
        if current_frames:
            markers[player_id].set_data([current_frames[-1]], [current_entropy[-1]])
            
            # Update label
            labels[player_id].set_position((current_frames[-1], current_entropy[-1]))
            labels[player_id].set_text(f"{player_id}\n{current_entropy[-1]:.2f}")
    
    return list(lines.values()) + list(markers.values()) + list(labels.values()) + [frame_text]

# Create animation
anim = animation.FuncAnimation(fig, update, init_func=init,
                             frames=len(frames), interval=100, blit=True)

# Adjust layout
plt.tight_layout(rect=[0, 0.05, 0.95, 0.95])

# Save animation
anim.save('entropy_graph.gif', writer='pillow', fps=10)
print("\nAnimation saved as 'entropy_graph.gif'") 