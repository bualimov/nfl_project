import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# PFF 2022 Team Defense Rankings (1 is best, 32 is worst)
pff_rankings = {
    'SF': 1, 'BUF': 2, 'DAL': 3, 'NE': 4, 'NYJ': 5,
    'BAL': 6, 'PHI': 7, 'DEN': 8, 'NO': 9, 'CIN': 10,
    'KC': 11, 'TB': 12, 'MIA': 13, 'CAR': 14, 'CLE': 15,
    'GB': 16, 'PIT': 17, 'JAX': 18, 'TEN': 19, 'LA': 20,
    'IND': 21, 'NYG': 22, 'DET': 23, 'SEA': 24, 'MIN': 25,
    'WAS': 26, 'HOU': 27, 'ATL': 28, 'LV': 29, 'LAC': 30,
    'ARI': 31, 'CHI': 32
}

def create_box_plot():
    # Set style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # Load team entropy data
    team_data = pd.read_csv('team_entropy_analysis.csv')
    
    # Create figure with specific size
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Sort teams by PFF ranking
    sorted_teams = sorted(pff_rankings.items(), key=lambda x: x[1])
    team_order = [team for team, _ in sorted_teams]
    
    # Create box plot data
    box_plot_data = []
    positions = []
    
    for team in team_order:
        team_row = team_data[team_data['team'] == team].iloc[0]
        box_data = {
            'median': team_row['median_entropy'],
            'q1': team_row['q1'],
            'q3': team_row['q3']
        }
        box_plot_data.append(box_data)
        positions.append(len(team_order) - pff_rankings[team])  # Reverse order for y-axis
    
    # Create horizontal box plots
    box_height = 0.6
    bar_color = '#0077be'  # Standard blue color like in DT plot
    
    for i, (pos, box_data) in enumerate(zip(positions, box_plot_data)):
        # Plot box
        plt.barh(pos, box_data['q3'] - box_data['q1'], 
                height=box_height,
                left=box_data['q1'],
                color=bar_color)
        
        # Plot median line
        plt.plot([box_data['median'], box_data['median']], 
                [pos - box_height/2, pos + box_height/2],
                color='white', linewidth=2)
        
        # Add team logo
        try:
            logo_path = f'logos/{team_order[i]}.png'
            if os.path.exists(logo_path):
                logo = Image.open(logo_path)
                logo_height = box_height * 0.8
                logo_box = plt.matplotlib.offsetbox.OffsetImage(logo, zoom=0.15)
                logo_box.image.axes = ax
                ab = plt.matplotlib.offsetbox.AnnotationBbox(
                    logo_box, (plt.xlim()[0] - 0.2, pos),
                    frameon=False, box_alignment=(1, 0.5)
                )
                ax.add_artist(ab)
        except Exception as e:
            print(f"Could not load logo for {team_order[i]}: {e}")
        
        # Add entropy value at end of bar
        plt.text(box_data['q3'] + 0.05, pos, 
                f"{box_data['median']:.1f}", 
                va='center', ha='left',
                fontsize=10)
    
    # Customize plot
    plt.title('NFL Team Defensive Entropy Distribution (2022)\nRanked by PFF Defense Rankings', 
              fontsize=16, pad=20, weight='bold')
    plt.xlabel('Defensive Entropy Value', fontsize=12, labelpad=10)
    
    # Set y-axis ticks and labels
    plt.yticks(positions, 
               [f"{i+1}. {team}" for i, team in enumerate(team_order)],
               fontsize=10)
    
    # Set background color and grid
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Set x-axis limits with padding
    x_min = min(box['q1'] for box in box_plot_data) - 0.5
    x_max = max(box['q3'] for box in box_plot_data) + 0.5
    plt.xlim(x_min, x_max)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig('team_entropy_rankings_2022.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'team_entropy_rankings_2022.png'")

if __name__ == "__main__":
    create_box_plot() 