import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os

# Load the results
df = pd.read_csv('mlb_player_entropy_analysis.csv')

def get_image(path):
    return OffsetImage(plt.imread(path), zoom=0.4)

# Set the style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Create figure and axis
fig, (ax_names, ax_bars) = plt.subplots(1, 2, figsize=(12, 12.5),
                                       gridspec_kw={'width_ratios': [0.8, 1.7]})

# Create y positions
y_positions = np.arange(len(df)*0.65, 0, -0.65)

# Add player names and logos in the left subplot
for i, (_, row) in enumerate(df.iterrows()):
    # Add rank and player name
    ax_names.text(0.2, y_positions[i], f"{i+1}. {row['displayName']}", 
                 fontsize=14, va='center', ha='left')

# Create horizontal bars
bars = ax_bars.barh(y_positions, df['normalized_entropy'], height=0.25, color='#1f77b4')

# Add entropy values at the end of each bar
for i, value in enumerate(df['normalized_entropy']):
    ax_bars.text(value + 0.1, y_positions[i], f"{value:.1f}", 
                va='center', ha='left', fontsize=14)

# Customize plots with two-line title
fig.text(0.5, 0.96, 'Middle Linebackers - 2022 Season', 
         fontsize=22, fontweight='bold', ha='center')
fig.text(0.5, 0.93, 'Ranked by Pre-Snap Movement Entropy Values', 
         fontsize=22, ha='center')

# Set axis labels
ax_bars.set_xlabel('Defensive Entropy Value', fontsize=14, fontweight='bold', labelpad=5)

# Set axis limits
ax_names.set_xlim(0, 1.1)
ax_bars.set_xlim(0, max(df['normalized_entropy']) + 5)

# Remove y-axis ticks and labels
ax_names.set_yticks([])
ax_bars.set_yticks([])

# Set y-axis limits
ax_names.set_ylim(-0.3, max(y_positions) + 0.3)
ax_bars.set_ylim(-0.3, max(y_positions) + 0.3)

# Remove spines from names subplot
ax_names.spines['top'].set_visible(False)
ax_names.spines['right'].set_visible(False)
ax_names.spines['bottom'].set_visible(False)
ax_names.spines['left'].set_visible(False)
ax_names.set_xticks([])

# Add grid to bars subplot
ax_bars.grid(True, axis='x', alpha=0.2)
ax_bars.tick_params(axis='x', labelsize=14)

# Make x-axis tick labels bold
for label in ax_bars.get_xticklabels():
    label.set_fontweight('bold')

# Adjust layout
plt.subplots_adjust(wspace=0, top=0.90, bottom=0.05, left=0.02, right=0.98)

# Save plot
plt.savefig('mlb_rankings_2022.png', dpi=300, bbox_inches='tight')
plt.close() 