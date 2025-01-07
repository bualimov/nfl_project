import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import seaborn as sns

def create_entropy_animation(tracking_df, play_df, output_file='entropy_animation.gif'):
    # Get events and frames
    events = tracking_df.sort_values('frameId')
    start_frame = events['frameId'].min()
    line_set_frame = events[events['event'] == 'line_set']['frameId'].min()
    snap_frame = events[events['event'] == 'ball_snap']['frameId'].min()
    
    print(f"Found frames - Start: {start_frame}, Line Set: {line_set_frame}, Snap: {snap_frame}")
    
    if pd.isna(line_set_frame) or pd.isna(snap_frame):
        print("Error: Could not find all required frames")
        return None
    
    # Initialize the figure with more height for titles
    fig = plt.figure(figsize=(10, 8))
    
    # Create main title at the top
    fig.suptitle('Defensive Player Entropy Over Time\nFrom First Frame to Snap', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    # Create subplot with specific position to control spacing
    ax = fig.add_axes([0.12, 0.15, 0.75, 0.65])  # [left, bottom, width, height]
    
    # Set up the plot
    ax.set_xlim(start_frame, snap_frame)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Normalized Entropy', fontsize=12)
    
    # Initialize lines dictionary with red tones
    lines = {}
    positions = ['CB', 'DE']  # Only CB and DE
    colors = ['#ff0000', '#8b0000']  # Red and Dark Red
    
    # Get player numbers for each position
    player_numbers = {}
    for pos in positions:
        pos_data = tracking_df[
            (tracking_df['frameId'] == start_frame) & 
            (tracking_df['position'] == pos)
        ].iloc[0]
        player_numbers[pos] = int(pos_data['jerseyNumber'])
    
    for pos, color in zip(positions, colors):
        # Create label with player number
        label = f"{pos} #{player_numbers[pos]}"
        line = ax.add_line(Line2D([], [], color=color, label=label, linewidth=2))
        lines[pos] = {'line': line, 'xdata': [], 'ydata': []}
    
    # Add vertical lines for events
    ax.axvline(x=start_frame, color='gray', linestyle='--', alpha=0.5)
    line_set_line = ax.axvline(x=line_set_frame, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=snap_frame, color='red', linestyle='--', alpha=0.5)
    
    # Add "Line Set" text annotation at the top of the plot
    ax.text(line_set_frame, 102, 'Line Set', 
            horizontalalignment='center', verticalalignment='bottom')
    
    # Create custom legend elements
    legend_elements = [
        Line2D([0], [0], color=color, label=f"{pos} #{player_numbers[pos]}", linewidth=2)
        for pos, color in zip(positions, colors)
    ]
    legend_elements.extend([
        Line2D([0], [0], color='gray', linestyle='--', label='Start', alpha=0.5),
        Line2D([0], [0], color='green', linestyle='--', label='Line Set', alpha=0.5),
        Line2D([0], [0], color='red', linestyle='--', label='Snap', alpha=0.5)
    ])
    
    # Add legend at the bottom
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=12)
    
    def init():
        for pos_data in lines.values():
            pos_data['line'].set_data([], [])
        return [pos_data['line'] for pos_data in lines.values()]
    
    def animate(frame):
        current_frame = frame + start_frame
        
        # Only calculate and show entropy between line_set and snap
        if line_set_frame <= current_frame <= snap_frame:
            frame_data = tracking_df[tracking_df['frameId'] == current_frame]
            
            for pos in positions:
                pos_data = frame_data[frame_data['position'] == pos]
                if not pos_data.empty:
                    # Take only the first player of each position
                    pos_data = pos_data.iloc[[0]]
                    entropy = calculate_entropy(pos_data)
                    lines[pos]['xdata'].append(current_frame)
                    lines[pos]['ydata'].append(entropy)
                    lines[pos]['line'].set_data(lines[pos]['xdata'], lines[pos]['ydata'])
        
        return [pos_data['line'] for pos_data in lines.values()]
    
    # Create animation
    frames = int(snap_frame - start_frame + 1)
    print(f"Creating animation with {frames} frames...")
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames,
                                 interval=100, blit=True)
    
    # Save animation
    print(f"Saving animation to {output_file}...")
    anim.save(output_file, writer='pillow')
    plt.close()
    print("Animation saved successfully")

def calculate_entropy(pos_data):
    """Calculate entropy for a player's position data"""
    # Constants
    w_theta = 0.3
    w_v = 0.2
    v_max = 10.0
    grid_size = 1.0
    sigma = 2.0
    
    # Get player data
    x = pos_data['x'].iloc[0]
    y = pos_data['y'].iloc[0]
    theta = pos_data['dir'].iloc[0]
    v = pos_data['s'].iloc[0]
    
    # Calculate angle relative to ball
    ball_y = 26.65  # Center of field width
    theta_relative = np.abs(theta - np.degrees(np.arctan2(ball_y - y, 60 - x)))
    
    # Create grid for field discretization
    x_grid = np.arange(max(0, x-10), min(120, x+10), grid_size)
    y_grid = np.arange(max(0, y-10), min(53.3, y+10), grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calculate base position probability
    p = np.zeros_like(X)
    for _, player in pos_data.iterrows():
        dist = np.sqrt((X - player['x'])**2 + (Y - player['y'])**2)
        p += np.exp(-dist**2/(2*sigma**2))
    
    # Normalize probabilities
    p = p / (np.sum(p) + 1e-10)
    
    # Calculate orientation factor
    orientation_factor = 1 + w_theta * np.cos(np.radians(theta_relative))
    
    # Calculate velocity factor
    velocity_factor = 1 + w_v * (min(v, v_max) / v_max)
    
    # Calculate entropy with modifiers
    base_entropy = -np.sum(p * np.log2(p + 1e-10))
    total_entropy = base_entropy * orientation_factor * velocity_factor
    
    # Normalize to 0-100 scale
    return total_entropy * 10  # Scale factor to get reasonable range

def main():
    print("Loading data...")
    # Load your data
    tracking_df = pd.read_csv('tracking_week_1.csv')
    plays_df = pd.read_csv('plays.csv')
    players_df = pd.read_csv('players.csv')
    
    # Get SEA vs DEN game
    print("Finding specific play...")
    sea_den_plays = plays_df[
        ((plays_df['possessionTeam'] == 'SEA') & (plays_df['defensiveTeam'] == 'DEN')) |
        ((plays_df['possessionTeam'] == 'DEN') & (plays_df['defensiveTeam'] == 'SEA'))
    ]
    sea_den_game = sea_den_plays['gameId'].iloc[0]
    
    # Filter for play 64 (a good example play)
    play_data = tracking_df[
        (tracking_df['gameId'] == sea_den_game) & 
        (tracking_df['playId'] == 64)
    ].copy()
    
    # Merge with players data to get positions
    play_data = play_data.merge(
        players_df[['nflId', 'position']], 
        on='nflId', 
        how='left'
    )
    
    play_info = plays_df[
        (plays_df['gameId'] == sea_den_game) & 
        (plays_df['playId'] == 64)
    ].iloc[0]
    
    print("Creating animation...")
    create_entropy_animation(play_data, play_info, 'entropy_graph.gif')
    print("Animation saved as 'entropy_graph.gif'")

if __name__ == "__main__":
    main() 