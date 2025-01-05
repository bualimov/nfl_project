import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

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
    
    # Set field view from end zone (0) to midfield (50)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 53.3)
    
    # Draw football field with more natural grass color
    field_color = '#2E8B57'  # Sea Green, more natural grass color
    ax.add_patch(Rectangle((0, 0), 100, 53.3, facecolor=field_color))
    
    # Add yard lines and numbers
    for yard in range(0, 51):  # Only up to 50 yard line
        # Yard lines
        if yard % 5 == 0:
            alpha = 0.5 if yard % 10 == 0 else 0.3
            ax.axvline(x=yard, color='white', linestyle='-', alpha=alpha)
            
            # Add hash marks at 5-yard intervals
            # NFL hash marks are at 23.36666 yards and 29.96666 yards from each sideline
            # Make hash marks shorter (0.5 yards) and use same alpha as yard lines
            ax.plot([yard - 0.25, yard + 0.25], [23.36666, 23.36666], 'w-', alpha=alpha)
            ax.plot([yard - 0.25, yard + 0.25], [29.96666, 29.96666], 'w-', alpha=alpha)
        
        # Yard numbers
        if yard % 10 == 0:
            if yard == 50:
                number = '50'
            else:
                number = str(yard)
            ax.text(yard, 5, number, ha='center', color='white', fontsize=16,
                   fontweight='bold', bbox=dict(facecolor='black', alpha=0.3,
                                              edgecolor='none', pad=0.5))
            # Add numbers on both sides
            ax.text(yard, 48.3, number, ha='center', color='white', fontsize=16,
                   fontweight='bold', bbox=dict(facecolor='black', alpha=0.3,
                                              edgecolor='none', pad=0.5))
    
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