import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ... [Previous code for entropy calculation and data loading remains the same] ...

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
        
        # Create line
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