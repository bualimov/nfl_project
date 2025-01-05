import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import math

# Read the tracking data for Week 1 and plays data
print("Reading tracking data...")
tracking_df = pd.read_csv('tracking_week_1.csv')
plays_df = pd.read_csv('plays.csv')

# Filter for play 64
play_df = tracking_df[tracking_df['playId'] == 64].copy()
game_id = play_df['gameId'].iloc[0]
play_details = plays_df[(plays_df['gameId'] == game_id) & (plays_df['playId'] == 64)].iloc[0]

# Sort by frame ID to ensure proper animation sequence
play_df = play_df.sort_values('frameId')

# Get the first frame to determine play direction and field position
first_frame = play_df[play_df['frameId'] == play_df['frameId'].min()]
play_direction = first_frame['playDirection'].iloc[0]

# Function to convert field coordinates
def convert_to_field_coords(x, absolute_yardline):
    if play_direction == 'right':
        return absolute_yardline + (x - 60)
    return 100 - (absolute_yardline + (x - 60))

# Get field position
absolute_yardline = play_details['absoluteYardlineNumber']
ball_start = convert_to_field_coords(60, absolute_yardline)

# Function to calculate player entropy
def calculate_entropy(x, y, all_positions):
    # Using a grid-based approach for entropy calculation
    grid_size = 1  # 1-yard grid
    sigma = 2.0    # Gaussian spread parameter
    
    # Create grid over relevant area
    x_grid = np.arange(max(0, x-10), min(120, x+10), grid_size)
    y_grid = np.arange(max(0, y-10), min(53.3, y+10), grid_size)
    
    # Calculate probability distribution
    p = np.zeros((len(x_grid), len(y_grid)))
    
    for pos_x, pos_y in all_positions:
        for i, gx in enumerate(x_grid):
            for j, gy in enumerate(y_grid):
                # Gaussian distribution around player position
                dist = np.sqrt((gx-pos_x)**2 + (gy-pos_y)**2)
                p[i,j] += np.exp(-dist**2/(2*sigma**2))
    
    # Normalize probabilities
    p = p / np.sum(p)
    
    # Calculate entropy
    entropy = -np.sum(p * np.log2(p + 1e-10))
    return entropy

# Create animation
def create_animation():
    # Set up the figure with two subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Add play description at the top
    play_text = (f"Week 1: {play_details['possessionTeam']} vs {play_details['defensiveTeam']} - "
                f"Q{play_details['quarter']} {play_details['gameClock']}\n"
                f"{play_details['playDescription']}\n"
                f"Formation: {play_details['offenseFormation']}, Coverage: {play_details['pff_passCoverage']}")
    plt.figtext(0.5, 0.95, play_text, ha='center', va='center', fontsize=12, wrap=True)
    
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    # Field plot
    ax1 = plt.subplot(gs[0])
    
    # Calculate visible field range (zoom in on the play)
    min_x = max(0, ball_start - 20)
    max_x = min(100, ball_start + 20)
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(0, 53.3)
    
    # Draw football field
    field_color = '#90EE90'  # Light green
    ax1.add_patch(Rectangle((0, 0), 100, 53.3, facecolor=field_color))
    
    # Add yard lines and numbers
    for yard in range(0, 101, 10):
        ax1.axvline(x=yard, color='white', linestyle='-', alpha=0.5)
        if yard == 50:
            number = '50'
        elif yard < 50:
            number = str(yard)
        else:
            number = str(100 - yard)
        ax1.text(yard, 2, number, ha='center', va='bottom', color='black', fontsize=10)
    
    # Add hash marks
    for yard in range(0, 101):
        ax1.plot([yard, yard], [0.5, 1.5], 'w-', alpha=0.3)
        ax1.plot([yard, yard], [51.8, 52.8], 'w-', alpha=0.3)
    
    # Remove axes
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Entropy plot
    ax2 = plt.subplot(gs[1])
    ax2.set_xlim(play_df['frameId'].min(), play_df['frameId'].max())
    ax2.set_ylim(0, 10)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Entropy')
    ax2.grid(True, alpha=0.3)
    
    # Initialize plots with larger markers
    offense_scatter = ax1.scatter([], [], c='blue', s=150)
    defense_scatter = ax1.scatter([], [], c='red', s=150)
    ball_scatter = ax1.scatter([], [], c='brown', s=80)
    
    # Initialize entropy history
    frames = sorted(play_df['frameId'].unique())
    entropy_history = {}
    frame_history = []
    
    # Create static text boxes for player labels
    player_labels = {}
    
    # Animation update function
    def update(frame):
        frame_data = play_df[play_df['frameId'] == frame].copy()
        frame_history.append(frame)
        
        # Update player positions
        offense_data = frame_data[frame_data['club'] == frame_data['club'].unique()[0]].copy()
        defense_data = frame_data[frame_data['club'] == frame_data['club'].unique()[1]].copy()
        ball_data = frame_data[frame_data['displayName'] == "football"].copy()
        
        # Convert coordinates
        offense_data.loc[:, 'x'] = offense_data.apply(lambda row: convert_to_field_coords(row['x'], absolute_yardline), axis=1)
        defense_data.loc[:, 'x'] = defense_data.apply(lambda row: convert_to_field_coords(row['x'], absolute_yardline), axis=1)
        if not ball_data.empty:
            ball_data.loc[:, 'x'] = ball_data.apply(lambda row: convert_to_field_coords(row['x'], absolute_yardline), axis=1)
        
        # Update scatter plots
        offense_scatter.set_offsets(np.c_[offense_data['x'], offense_data['y']])
        defense_scatter.set_offsets(np.c_[defense_data['x'], defense_data['y']])
        if not ball_data.empty:
            ball_scatter.set_offsets(np.c_[ball_data['x'], ball_data['y']])
        
        # Calculate entropy for defensive players
        defense_positions = list(zip(defense_data['x'], defense_data['y']))
        
        # Update player labels
        for _, player in frame_data.iterrows():
            if player['displayName'] != "football":
                player_id = player['displayName']
                x = convert_to_field_coords(player['x'], absolute_yardline)
                y = player['y']
                
                if player['club'] == frame_data['club'].unique()[1]:  # Defensive player
                    entropy = calculate_entropy(x, y, defense_positions)
                    text = f"DEF\nE: {entropy:.2f}"
                    
                    # Store entropy for plotting
                    if player_id not in entropy_history:
                        entropy_history[player_id] = {'values': [], 'line': None}
                    entropy_history[player_id]['values'].append(entropy)
                else:
                    text = "OFF"
                
                if player_id not in player_labels:
                    player_labels[player_id] = ax1.text(x, y, text,
                                                      ha='center', va='bottom',
                                                      bbox=dict(facecolor='white',
                                                              edgecolor='none',
                                                              alpha=0.7))
                else:
                    player_labels[player_id].set_position((x, y))
                    player_labels[player_id].set_text(text)
        
        # Update entropy plot
        for player_id in entropy_history:
            values = entropy_history[player_id]['values']
            if values:
                if entropy_history[player_id]['line'] is None:
                    entropy_history[player_id]['line'], = ax2.plot(frame_history[-len(values):],
                                                                 values, alpha=0.5)
                else:
                    entropy_history[player_id]['line'].set_data(frame_history[-len(values):],
                                                              values)
        
        return [offense_scatter, defense_scatter, ball_scatter]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                 interval=50, blit=True)
    
    # Save animation
    anim.save('entropy_animation.gif', writer='pillow', fps=20)
    print("\nAnimation saved as 'entropy_animation.gif'")

# Generate the animation
create_animation() 