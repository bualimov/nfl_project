import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import matplotlib.patheffects as path_effects

# Read the tracking data, plays data, and players data
print("Reading data...")
tracking_df = pd.read_csv('tracking_week_1.csv')
plays_df = pd.read_csv('plays.csv')
players_df = pd.read_csv('players.csv')

# First, find the SEA vs DEN game from Week 1
sea_den_plays = plays_df[
    ((plays_df['possessionTeam'] == 'SEA') & (plays_df['defensiveTeam'] == 'DEN')) |
    ((plays_df['possessionTeam'] == 'DEN') & (plays_df['defensiveTeam'] == 'SEA'))
]
sea_den_game = sea_den_plays['gameId'].iloc[0]

print(f"\nFound SEA vs DEN game ID: {sea_den_game}")

# Filter tracking data for this game and play 64
play_df = tracking_df[
    (tracking_df['gameId'] == sea_den_game) & 
    (tracking_df['playId'] == 64)
].copy()

# Get play details
play_details = plays_df[
    (plays_df['gameId'] == sea_den_game) & 
    (plays_df['playId'] == 64)
].iloc[0]

# Verify it's the correct game and play
possession_team = play_details['possessionTeam']
defensive_team = play_details['defensiveTeam']
print(f"Analyzing play 64: {possession_team} vs {defensive_team}")
print(f"Play description: {play_details['playDescription']}")

# Join with players data to get positions
play_df = play_df.merge(players_df[['nflId', 'position']], 
                       on='nflId', 
                       how='left')

# Find snap frame and filter for pre-snap frames
snap_frame = play_df[play_df['event'] == 'ball_snap']['frameId'].iloc[0]
play_df = play_df[play_df['frameId'] <= snap_frame].copy()

# Sort by frame ID
play_df = play_df.sort_values('frameId')

# Get play direction
play_direction = play_df['playDirection'].iloc[0]

# Function to convert field coordinates
def convert_to_field_coords(x, absolute_yardline):
    if play_direction == 'right':
        return absolute_yardline + (x - 60)
    return 100 - (absolute_yardline + (x - 60))

# Get field position
absolute_yardline = play_details['absoluteYardlineNumber']
ball_start = convert_to_field_coords(60, absolute_yardline)

# NFL team colors
TEAM_COLORS = {
    'SEA': '#002244',  # Seahawks Navy Blue
    'DEN': '#FB4F14'   # Broncos Orange
}

def create_animation():
    # Set up the figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Add play description at the top
    play_text = (f"NFL Week 1, 2022: {possession_team} vs {defensive_team}\n"
                f"Q{play_details['quarter']} {play_details['gameClock']} - {play_details['down']} & {play_details['yardsToGo']}\n"
                f"{play_details['playDescription']}")
    plt.suptitle(play_text, y=0.95, fontsize=12, wrap=True)
    
    # Calculate field view to ensure all players are visible
    frame_data = play_df[play_df['frameId'] == snap_frame].copy()
    player_x_coords = frame_data.apply(lambda row: convert_to_field_coords(row['x'], absolute_yardline), axis=1)
    min_player_x = player_x_coords.min()
    max_player_x = player_x_coords.max()
    
    # Add padding to ensure all players are visible
    field_padding = 5  # yards of padding on each side
    min_x = max(0, min(min_player_x - field_padding, ball_start - 25))
    max_x = min(100, max(max_player_x + field_padding, ball_start + 25))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(0, 53.3)
    
    # Draw football field with original green
    field_color = '#90EE90'
    ax.add_patch(Rectangle((0, 0), 100, 53.3, facecolor=field_color))
    
    # Add Broncos logo at the 50-yard line
    logo_circle = Circle((50, 26.65), radius=6, facecolor=TEAM_COLORS['DEN'], alpha=0.3)
    ax.add_patch(logo_circle)
    
    # Add "DEN" text in the circle
    logo_text = ax.text(50, 26.65, 'DEN', color='white', fontsize=20, fontweight='bold',
                       ha='center', va='center')
    logo_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
    
    # Add yard lines and numbers
    for yard in range(0, 101):
        # Yard lines
        if yard % 5 == 0:
            alpha = 0.5 if yard % 10 == 0 else 0.3
            ax.axvline(x=yard, color='white', linestyle='-', alpha=alpha)
        
        # Yard numbers
        if yard % 10 == 0:
            if yard == 50:
                number = '50'
            elif yard < 50:
                number = str(yard)
            else:
                number = str(100 - yard)
            ax.text(yard, 5, number, ha='center', color='white', fontsize=16,
                   fontweight='bold', bbox=dict(facecolor='black', alpha=0.3,
                                              edgecolor='none', pad=0.5))
    
    # Add hash marks
    for yard in range(0, 101):
        ax.plot([yard, yard], [0.5, 1.5], 'w-', alpha=0.3)
        ax.plot([yard, yard], [51.8, 52.8], 'w-', alpha=0.3)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Initialize scatter plots with larger markers
    offense_scatter = ax.scatter([], [], c=TEAM_COLORS[possession_team], s=250, 
                               label=f'Offense ({possession_team})')
    defense_scatter = ax.scatter([], [], c=TEAM_COLORS[defensive_team], s=250, 
                               label=f'Defense ({defensive_team})')
    ball_scatter = ax.scatter([], [], c='brown', s=100, label='Football')
    
    # Create static text boxes for player labels
    player_labels = {}
    
    # Add frame counter and event text
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top')
    event_text = ax.text(0.02, 0.94, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', color='red')
    
    # Animation update function
    def update(frame):
        frame_data = play_df[play_df['frameId'] == frame].copy()
        
        # Update frame counter
        frame_text.set_text(f'Frame: {frame}')
        
        # Update event text
        events = frame_data['event'].unique()
        event = next((e for e in events if pd.notna(e)), '')
        event_text.set_text(f'Event: {event}')
        
        # Update player positions
        offense_data = frame_data[frame_data['club'] == possession_team].copy()
        defense_data = frame_data[frame_data['club'] == defensive_team].copy()
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
        
        # Update player labels
        for _, player in frame_data.iterrows():
            if player['displayName'] != "football":
                player_id = player['displayName']
                x = convert_to_field_coords(player['x'], absolute_yardline)
                y = player['y']
                
                # Use position and jersey number (as integer)
                text = f"{player['position']}\n#{int(player['jerseyNumber'])}"
                
                if player_id not in player_labels:
                    player_labels[player_id] = ax.text(x, y, text,
                                                     ha='center', va='center',
                                                     color='white', fontweight='bold',
                                                     fontsize=8,
                                                     bbox=dict(facecolor='black', alpha=0.3,
                                                             edgecolor='none', pad=0.5))
                else:
                    player_labels[player_id].set_position((x, y))
                    player_labels[player_id].set_text(text)
        
        return [offense_scatter, defense_scatter, ball_scatter, frame_text, event_text] + list(player_labels.values())
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Create animation
    frames = sorted(play_df['frameId'].unique())
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                 interval=100, blit=True)
    
    # Save animation
    anim.save('play_animation.gif', writer='pillow', fps=10)
    print("\nAnimation saved as 'play_animation.gif'")

# Generate the animation
create_animation() 