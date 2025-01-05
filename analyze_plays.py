import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    # Read plays data
    print("Reading plays data...")
    plays_df = pd.read_csv('plays.csv')

    # Get unique plays (gameId, playId combinations)
    unique_plays = plays_df[['gameId', 'playId']].drop_duplicates()
    total_plays = len(unique_plays)

    # Count plays with penalties
    plays_with_penalties = plays_df[plays_df['playNullifiedByPenalty'] == True][['gameId', 'playId']].drop_duplicates()
    penalty_count = len(plays_with_penalties)
    clean_plays = total_plays - penalty_count

    print("\nPlay Counts:")
    print(f"Total unique plays: {total_plays}")
    print(f"Plays with penalties: {penalty_count}")
    print(f"Plays without penalties: {clean_plays}")

    # Analyze formations and coverages for clean plays
    clean_plays_df = plays_df[~plays_df[['gameId', 'playId']].apply(tuple, axis=1).isin(
        plays_with_penalties[['gameId', 'playId']].apply(tuple, axis=1)
    )]

    # Set figure style
    plt.rcParams['font.size'] = 14
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Color schemes
    blue_colors = ['#1f77b4', '#3498db', '#5dade2', '#85c1e9', '#aed6f1', '#d6eaf8', '#ebf5fb']
    red_colors = ['#c0392b', '#e74c3c', '#ec7063', '#f1948a', '#f5b7b1', '#fadbd8', '#fdedec']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 18))  # Increased height to accommodate title
    
    # Add total plays text at the top
    fig.suptitle(f'Total Plays Analyzed: {total_plays:,}', 
                 y=0.95, fontsize=20, weight='bold')
    
    # 1. Offensive Formation and Receiver Alignment Chart
    plt.subplot(2, 1, 1)
    
    # Create cross-tabulation of formations and alignments
    off_cross = pd.crosstab(clean_plays_df['offenseFormation'], clean_plays_df['receiverAlignment'])
    
    # Sort formations by total frequency
    formation_totals = off_cross.sum(axis=1)
    off_cross = off_cross.loc[formation_totals.sort_values(ascending=False).index]
    
    # Create stacked bar chart
    bottom = np.zeros(len(off_cross))
    bars = []
    min_segment_height = 100  # Minimum height for showing labels
    
    for i, alignment in enumerate(off_cross.columns):
        bars.append(plt.bar(off_cross.index, off_cross[alignment], bottom=bottom, 
                label=f'{alignment}', color=blue_colors[i % len(blue_colors)],
                edgecolor='black', linewidth=1))
        # Add labels for significant segments
        for j, value in enumerate(off_cross[alignment]):
            if value > min_segment_height:
                center_y = bottom[j] + value/2
                plt.text(j, center_y, alignment,
                        fontsize=12,
                        weight='bold',
                        color='black',
                        ha='center',
                        va='center',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))
        bottom += off_cross[alignment]
    
    plt.title('Offensive Formation and Receiver Alignment Distribution', pad=20, size=20, weight='bold')
    plt.xlabel('Offensive Formation', size=16, weight='bold')
    plt.ylabel('Number of Plays', size=16, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.legend(title='Receiver Alignment', bbox_to_anchor=(1.05, 1), loc='upper left', 
              title_fontsize=14, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 9000)  # Set consistent y-axis limit

    # 2. Defensive Coverage and Man/Zone Chart
    plt.subplot(2, 1, 2)
    
    # Create cross-tabulation of coverage and man/zone
    def_cross = pd.crosstab(clean_plays_df['pff_passCoverage'], clean_plays_df['pff_manZone'])
    
    # Sort coverages by total frequency and keep top 8
    coverage_totals = def_cross.sum(axis=1)
    top_coverages = coverage_totals.sort_values(ascending=False).head(8)
    def_cross = def_cross.loc[top_coverages.index]
    
    # Create vertical stacked bar chart
    bottom = np.zeros(len(def_cross))
    for i, coverage_type in enumerate(def_cross.columns):
        plt.bar(def_cross.index, def_cross[coverage_type], bottom=bottom,
               label=f'{coverage_type}', color=red_colors[i % len(red_colors)],
               edgecolor='black', linewidth=1)
        # Add labels for significant segments
        for j, value in enumerate(def_cross[coverage_type]):
            if value > min_segment_height:
                center_y = bottom[j] + value/2
                plt.text(j, center_y, coverage_type,
                        fontsize=12,
                        weight='bold',
                        color='black',
                        ha='center',
                        va='center',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))
        bottom += def_cross[coverage_type]
    
    plt.title('Defensive Coverage and Man/Zone Distribution', pad=20, size=20, weight='bold')
    plt.xlabel('Coverage Type', size=16, weight='bold')
    plt.ylabel('Number of Plays', size=16, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.legend(title='Coverage Scheme', bbox_to_anchor=(1.05, 1), loc='upper left',
              title_fontsize=14, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 9000)  # Set consistent y-axis limit

    # Adjust layout and save
    plt.tight_layout(pad=4.0, rect=[0, 0, 0.85, 0.95])  # Make room for legends
    plt.savefig('formation_coverage_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'formation_coverage_analysis.png'")

    # Print text summaries
    print("\nOffensive Formation Breakdown:")
    print(formation_totals.to_string())
    
    print("\nDefensive Coverage Breakdown:")
    print(coverage_totals.to_string())

except Exception as e:
    print(f"An error occurred: {str(e)}") 