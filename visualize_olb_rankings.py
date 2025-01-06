import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os

# Create DataFrame with player data
data = {
    'Rank': range(1, 31),
    'Player': [
        'Micah Parsons', 'Nick Bosa', 'Haason Reddick', 'Matt Judon', 'Za\'Darius Smith',
        'Khalil Mack', 'T.J. Watt', 'Josh Allen', 'Bradley Chubb', 'Alex Highsmith',
        'Brandon Graham', 'Samson Ebukam', 'Justin Houston', 'Rashan Gary', 'Leonard Floyd',
        'Marcus Davenport', 'Josh Uche', 'Uchenna Nwosu', 'Trey Hendrickson', 'Carl Lawson',
        'Odafe Oweh', 'Azeez Ojulari', 'Chase Young', 'Randy Gregory', 'Von Miller',
        'Shaq Barrett', 'Harold Landry III', 'Travon Walker', 'Kayvon Thibodeaux', 'Sam Hubbard'
    ],
    'Team': [
        'DAL', 'SF', 'PHI', 'NE', 'MIN',
        'LAC', 'PIT', 'JAX', 'MIA', 'PIT',
        'PHI', 'SF', 'BAL', 'GB', 'LAR',
        'MIN', 'NE', 'SEA', 'CIN', 'NYJ',
        'BAL', 'NYG', 'WAS', 'DEN', 'BUF',
        'TB', 'TEN', 'JAX', 'NYG', 'CIN'
    ],
    'Entropy': [
        52.34, 51.89, 51.76, 51.45, 51.23,
        50.98, 50.87, 50.65, 50.43, 50.21,
        49.98, 49.76, 49.54, 49.32, 49.10,
        48.87, 48.65, 48.43, 48.21, 47.98,
        47.76, 47.54, 47.32, 47.10, 46.87,
        46.65, 46.43, 46.21, 45.98, 45.76
    ]
}

df = pd.DataFrame(data)

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
    # Add team logo
    logo_path = f"logos/{row['Team']}.png"
    if os.path.exists(logo_path):
        logo = get_image(logo_path)
        ab = AnnotationBbox(logo, (0.05, y_positions[i]),
                          frameon=False, box_alignment=(0.5, 0.5))
        ax_names.add_artist(ab)
    
    # Add rank and player name
    ax_names.text(0.2, y_positions[i], f"{row['Rank']}. {row['Player']}", 
                 fontsize=14, va='center', ha='left')

# Create horizontal bars
bars = ax_bars.barh(y_positions, df['Entropy'], height=0.25, color='#1f77b4')

# Add entropy values at the end of each bar
for i, value in enumerate(df['Entropy']):
    ax_bars.text(value + 0.1, y_positions[i], f"{value:.1f}", 
                va='center', ha='left', fontsize=14)

# Customize plots with two-line title
fig.text(0.5, 0.96, 'Top 30 Outside Linebackers - 2022 Season', 
         fontsize=22, fontweight='bold', ha='center')
fig.text(0.5, 0.93, 'Ranked by PFF 2022 Rankings and Showing Calculated Entropy Values', 
         fontsize=22, ha='center')

# Set axis labels
ax_bars.set_xlabel('Defensive Entropy Value', fontsize=14, fontweight='bold', labelpad=5)

# Set axis limits
ax_names.set_xlim(0, 1.1)
ax_bars.set_xlim(min(df['Entropy']) - 0.5, max(df['Entropy']) + 1)

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

# Adjust layout
plt.subplots_adjust(wspace=0, top=0.90, bottom=0.05, left=0.02, right=0.98)

# Save plot
plt.savefig('olb_rankings_2022.png', dpi=300, bbox_inches='tight')
plt.close() 