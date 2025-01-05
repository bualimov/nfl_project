import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import seaborn as sns

def create_combined_animation(tracking_df, play_df, output_file='combined_animation.gif'):
    # Get events and frames
    events = tracking_df.sort_values('frameId')
    start_frame = events['frameId'].min()
    line_set_frame = events[events['event'] == 'line_set']['frameId'].min()
    snap_frame = events[events['event'] == 'ball_snap']['frameId'].min()
    
    # Initialize the figure with space for both plots
    fig = plt.figure(figsize=(10, 14))
    
    # Create main title with play details
    title_text = (f"Play {int(play_df['playId'])} - Week 1 - {play_df['possessionTeam']} vs {play_df['defensiveTeam']}")
    fig.text(0.5, 0.98, title_text, fontsize=12, fontweight='bold', ha='center')
    
    # Add subtitle lines without bold - reduced spacing between lines
    fig.text(0.5, 0.965, f"Q{play_df['quarter']} {play_df['gameClock']}", 
             fontsize=12, ha='center')
    fig.text(0.5, 0.95, play_df['playDescription'], 
             fontsize=12, ha='center')
    
    # Create subplots with consistent width
    field_ax = fig.add_axes([0.12, 0.48, 0.75, 0.4])  # Field plot position
    graph_ax = fig.add_axes([0.12, 0.15, 0.75, 0.25])  # Bottom plot position unchanged
    
    # Add frame counter and event text above field with less spacing
    frame_text = fig.text(0.12, 0.90, '', fontsize=10, ha='left')
    event_text = fig.text(0.12, 0.885, '', fontsize=10, ha='left')
    
    # Set up field plot - show 0 to 50 yard line
    field_ax.set_xlim(10, 60)  # Show only 0-50 yards (tracking coords 10-60)
    field_ax.set_ylim(0, 53.3)
    field_ax.set_aspect('auto')
    
    # Add padding to maintain field proportions
    field_ax.set_position([0.12, 0.48, 0.75, 0.4])  # Moved up to match creation
    
    # Draw field markings
    field_ax.set_facecolor('#2E8B57')  # Sea Green - more natural grass color
    field_ax.grid(False)  # Remove horizontal grid
    
    # Hide default axis markings
    field_ax.set_xticks([])
    field_ax.set_yticks([])
    
    # Add yard lines and numbers (converting from tracking coords to actual yards)
    for yard in range(10, 60, 5):  # Start at 0 yard line (10 in tracking coords)
        actual_yard = yard - 10  # Convert to actual yard line number
        alpha = 1.0 if yard % 10 == 0 else 0.3
        field_ax.axvline(yard, color='white', linestyle='-', alpha=alpha)
        if yard % 10 == 0 and actual_yard > 0:  # Only show numbers on 10-yard lines, skip 0
            # Add background box and larger text at bottom
            bbox_props = dict(boxstyle='round,pad=0.3', fc='#2E8B57', alpha=0.7, ec='#2E8B57')  # Match field color
            field_ax.text(yard, 5, str(actual_yard), 
                         ha='center', va='center', 
                         color='white', 
                         fontsize=16,
                         fontweight='bold',
                         bbox=bbox_props)
            # Add background box and larger text at top
            field_ax.text(yard, 48, str(actual_yard), 
                         ha='center', va='center', 
                         color='white', 
                         fontsize=16,
                         fontweight='bold',
                         bbox=bbox_props)
    
    # Set up entropy plot
    graph_ax.set_xlim(start_frame, snap_frame)
    graph_ax.set_ylim(0, 100)
    graph_ax.set_xlabel('Frame', fontsize=12)
    graph_ax.set_ylabel('Normalized Entropy', fontsize=12)
    graph_ax.grid(True, alpha=0.3)
    
    # Initialize scatter plots for players
    positions = ['CB', 'DE']  # Only CB and DE
    colors = ['#ff0000', '#8b0000']  # Red and Dark Red
    
    # Use exact NFL team colors
    team_colors = {
        'SEA': '#002244',  # Seahawks Navy
        'DEN': '#FB4F14'   # Broncos Orange
    }
    home_team = play_df['defensiveTeam']
    away_team = play_df['possessionTeam']
    home_color = team_colors[home_team]
    away_color = team_colors[away_team]
    
    # Add team color legend at bottom of field plot
    team_legend = [
        plt.Rectangle((0, 0), 1, 1, fc=home_color, alpha=0.8, label=f'{home_team} (Defense)'),
        plt.Rectangle((0, 0), 1, 1, fc=away_color, alpha=0.8, label=f'{away_team} (Offense)')
    ]
    field_ax.legend(handles=team_legend, loc='upper center', 
                   bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=10)  # Moved closer to field
    
    # Initialize player plots, labels, and entropy lines
    players = {}
    player_labels = {}
    lines = {}
    all_player_labels = {}
    
    # Initialize highlighted players
    for pos, color in zip(positions, colors):
        scatter = field_ax.scatter([], [], c=color, label=f"{pos}", s=400, zorder=3)
        players[pos] = scatter
        label = field_ax.text(0, 0, "", ha='center', va='center', color='white',
                            fontsize=8, fontweight='normal')
        player_labels[pos] = label
        
        line = graph_ax.add_line(Line2D([], [], color=color, label=f"{pos}", linewidth=2))
        lines[pos] = {'line': line, 'xdata': [], 'ydata': []}
    
    # Add other players
    home_players = field_ax.scatter([], [], c=home_color, alpha=0.8, s=400, zorder=2)
    away_players = field_ax.scatter([], [], c=away_color, alpha=0.8, s=400, zorder=2)
    
    # Add ball
    ball = field_ax.scatter([], [], c='brown', marker='o', s=100, zorder=4)
    
    # Add vertical lines for events in entropy plot
    graph_ax.axvline(x=start_frame, color='gray', linestyle='--', alpha=0.5)
    graph_ax.axvline(x=line_set_frame, color='green', linestyle='--', alpha=0.5)
    graph_ax.axvline(x=snap_frame, color='red', linestyle='--', alpha=0.5)
    
    # Add "Line Set" text annotation
    graph_ax.text(line_set_frame, 102, 'Line Set', 
                 horizontalalignment='center', verticalalignment='bottom')
    
    # Create legend for entropy plot
    legend_elements = [
        Line2D([0], [0], color=color, label=f"{pos}", linewidth=2)
        for pos, color in zip(positions, colors)
    ]
    legend_elements.extend([
        Line2D([0], [0], color='gray', linestyle='--', label='Start', alpha=0.5),
        Line2D([0], [0], color='green', linestyle='--', label='Line Set', alpha=0.5),
        Line2D([0], [0], color='red', linestyle='--', label='Snap', alpha=0.5)
    ])
    
    # Add legend at the bottom with minimal spacing
    graph_ax.legend(handles=legend_elements, loc='upper center', 
                   bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=10)
    
    def init():
        # Initialize all elements
        for scatter in players.values():
            scatter.set_offsets(np.c_[[], []])
        for label in {**player_labels, **all_player_labels}.values():
            label.set_position((0, 0))
            label.set_text("")
        home_players.set_offsets(np.c_[[], []])
        away_players.set_offsets(np.c_[[], []])
        ball.set_offsets(np.c_[[], []])
        frame_text.set_text("")
        event_text.set_text("")
        
        for pos_data in lines.values():
            pos_data['line'].set_data([], [])
        
        return ([*players.values(), *player_labels.values(), *all_player_labels.values(),
                home_players, away_players, ball] + 
                [pos_data['line'] for pos_data in lines.values()])
    
    def animate(frame):
        current_frame = frame + start_frame
        frame_data = tracking_df[tracking_df['frameId'] == current_frame]
        
        # Update frame counter and event text
        frame_text.set_text(f'Frame: {current_frame}')
        event = ""
        if current_frame == line_set_frame:
            event = "Event: Line Set"
        elif current_frame == snap_frame:
            event = "Event: Snap"
        event_text.set_text(event)
        
        # Update all players and their labels
        home_data = frame_data[frame_data['club'] == home_team]
        away_data = frame_data[frame_data['club'] == away_team]
        
        # Update defensive players
        home_players.set_offsets(np.c_[home_data['x'], home_data['y']])
        for _, player in home_data.iterrows():
            player_id = f"{player['position']}_{player['jerseyNumber']}"
            if player_id not in all_player_labels:
                label = field_ax.text(0, 0, "", ha='center', va='center',
                                    color='white', fontsize=8, fontweight='normal')
                all_player_labels[player_id] = label
            label = all_player_labels[player_id]
            label.set_position((player['x'], player['y']))
            label.set_text(f"{player['position']}\n{int(player['jerseyNumber'])}")
        
        # Update offensive players
        away_players.set_offsets(np.c_[away_data['x'], away_data['y']])
        for _, player in away_data.iterrows():
            player_id = f"{player['position']}_{player['jerseyNumber']}"
            if player_id not in all_player_labels:
                label = field_ax.text(0, 0, "", ha='center', va='center',
                                    color='white', fontsize=8, fontweight='normal')
                all_player_labels[player_id] = label
            label = all_player_labels[player_id]
            label.set_position((player['x'], player['y']))
            label.set_text(f"{player['position']}\n{int(player['jerseyNumber'])}")
        
        # Update highlighted players
        for pos in positions:
            pos_data = frame_data[frame_data['position'] == pos]
            if not pos_data.empty:
                pos_data = pos_data.iloc[[0]]
                x, y = pos_data['x'].iloc[0], pos_data['y'].iloc[0]
                players[pos].set_offsets(np.c_[[x], [y]])
                player_labels[pos].set_position((x, y))
                player_labels[pos].set_text(f"{pos}\n{int(pos_data['jerseyNumber'].iloc[0])}")
        
        # Update ball position
        ball_data = frame_data[frame_data['club'] == 'football']
        if not ball_data.empty:
            ball.set_offsets(np.c_[ball_data['x'], ball_data['y']])
        
        # Update entropy lines
        if line_set_frame <= current_frame <= snap_frame:
            for pos in positions:
                pos_data = frame_data[frame_data['position'] == pos]
                if not pos_data.empty:
                    pos_data = pos_data.iloc[[0]]
                    entropy = calculate_entropy(pos_data)
                    lines[pos]['xdata'].append(current_frame)
                    lines[pos]['ydata'].append(entropy)
                    lines[pos]['line'].set_data(lines[pos]['xdata'], lines[pos]['ydata'])
        
        return ([*players.values(), *player_labels.values(), *all_player_labels.values(),
                home_players, away_players, ball] + 
                [pos_data['line'] for pos_data in lines.values()])
    
    # Create animation
    frames = int(snap_frame - start_frame + 1)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames,
                                 interval=100, blit=True)
    
    # Save animation
    anim.save(output_file, writer='pillow')
    plt.close()

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
    
    return total_entropy * 10  # Scale factor to get reasonable range

def main():
    print("Loading data...")
    tracking_df = pd.read_csv('tracking_week_1.csv')
    plays_df = pd.read_csv('plays.csv')
    players_df = pd.read_csv('players.csv')
    
    print("Finding specific play...")
    sea_den_plays = plays_df[
        ((plays_df['possessionTeam'] == 'SEA') & (plays_df['defensiveTeam'] == 'DEN')) |
        ((plays_df['possessionTeam'] == 'DEN') & (plays_df['defensiveTeam'] == 'SEA'))
    ]
    sea_den_game = sea_den_plays['gameId'].iloc[0]
    
    play_data = tracking_df[
        (tracking_df['gameId'] == sea_den_game) & 
        (tracking_df['playId'] == 64)
    ].copy()
    
    play_data = play_data.merge(
        players_df[['nflId', 'position']], 
        on='nflId', 
        how='left'
    )
    
    play_info = plays_df[
        (plays_df['gameId'] == sea_den_game) & 
        (plays_df['playId'] == 64)
    ].iloc[0]
    
    print("Creating combined animation...")
    create_combined_animation(play_data, play_info)
    print("Animation saved as 'combined_animation.gif'")

if __name__ == "__main__":
    main() 