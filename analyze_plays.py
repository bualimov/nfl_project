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
    plt.rcParams['font.size'] = 14  # Increased base font size
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Color schemes - previous colors
    blue_colors = ['#1f77b4', '#3498db', '#5dade2', '#85c1e9', '#aed6f1', '#d6eaf8', '#ebf5fb']
    red_colors = ['#c0392b', '#e74c3c', '#ec7063', '#f1948a', '#f5b7b1', '#fadbd8', '#fdedec']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Add total plays text at the top
    plt.figtext(0.5, 0.95, f'Total Plays Analyzed: {total_plays:,}', 
                ha='center', va='center', fontsize=20, weight='bold')
    
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
    for i, alignment in enumerate(off_cross.columns):
        bars.append(plt.bar(off_cross.index, off_cross[alignment], bottom=bottom, 
                label=f'{alignment}', color=blue_colors[i % len(blue_colors)],
                edgecolor='black', linewidth=1))
        # Add value labels on the bars
        for j, value in enumerate(off_cross[alignment]):
            if value > 0:  # Only show labels for non-zero values
                plt.text(j, bottom[j] + value/2, str(value),
                        ha='center', va='center', fontsize=12, color='black')
        bottom += off_cross[alignment]
    
    # Add total counts and percentages above bars
    for i, (formation, total) in enumerate(formation_totals.items()):
        percentage = total / total_plays * 100
        plt.text(i, total + 200, f'{total:,}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=14, weight='bold')
    
    plt.title('Offensive Formation and Receiver Alignment Distribution', pad=20, size=20, weight='bold')
    plt.xlabel('Offensive Formation', size=16, weight='bold')
    plt.ylabel('Number of Plays', size=16, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.legend(title='Receiver Alignment', bbox_to_anchor=(1.05, 1), loc='upper left', 
              title_fontsize=14, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Defensive Coverage and Man/Zone Chart
    plt.subplot(2, 1, 2)
    
    # Create cross-tabulation of coverage and man/zone
    def_cross = pd.crosstab(clean_plays_df['pff_passCoverage'], clean_plays_df['pff_manZone'])
    
    # Sort coverages by total frequency and keep top 8
    coverage_totals = def_cross.sum(axis=1)
    top_coverages = coverage_totals.sort_values(ascending=True).tail(8)
    def_cross = def_cross.loc[top_coverages.index]
    
    # Create horizontal stacked bar chart
    left = np.zeros(len(def_cross))
    for i, coverage_type in enumerate(def_cross.columns):
        bars = plt.barh(def_cross.index, def_cross[coverage_type], left=left,
                label=f'{coverage_type}', color=red_colors[i % len(red_colors)],
                edgecolor='black', linewidth=1)
        # Add value labels on the bars
        for j, value in enumerate(def_cross[coverage_type]):
            if value > 0:  # Only show labels for non-zero values
                plt.text(left[j] + value/2, j, str(value),
                        ha='center', va='center', fontsize=12, color='black')
        left += def_cross[coverage_type]
    
    # Add total counts and percentages at end of bars
    for i, (coverage, total) in enumerate(def_cross.sum(axis=1).items()):
        percentage = total / total_plays * 100
        plt.text(total + 100, i, f'{total:,} ({percentage:.1f}%)', 
                va='center', fontsize=14, weight='bold')
    
    plt.title('Defensive Coverage and Man/Zone Distribution', pad=20, size=20, weight='bold')
    plt.xlabel('Number of Plays', size=16, weight='bold')
    plt.ylabel('Coverage Type', size=16, weight='bold')
    plt.legend(title='Coverage Scheme', bbox_to_anchor=(1.05, 1), loc='upper left',
              title_fontsize=14, fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=14)

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