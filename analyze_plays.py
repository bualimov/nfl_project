import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    # Read all tracking data files
    print("Reading tracking data...")
    all_tracking_data = []
    for week in range(1, 10):  # Weeks 1-9
        df = pd.read_csv(f'tracking_week_{week}.csv')
        all_tracking_data.append(df)
    
    tracking_df = pd.concat(all_tracking_data)
    
    # Read plays data
    print("Reading plays data...")
    plays_df = pd.read_csv('plays.csv')
    
    # First, identify valid plays based on snap timing
    print("\nIdentifying valid plays based on snap timing...")
    valid_plays = []
    for (game_id, play_id), play_data in tracking_df.groupby(['gameId', 'playId']):
        # Get events in chronological order
        events = play_data.sort_values('frameId')['event'].dropna().unique()
        
        # Find line_set and ball_snap events
        line_set_frame = play_data[play_data['event'] == 'line_set']['frameId'].min()
        snap_frame = play_data[play_data['event'] == 'ball_snap']['frameId'].min()
        
        # If both events exist and in correct order, calculate time difference
        if (pd.notna(line_set_frame) and pd.notna(snap_frame) and 
            snap_frame > line_set_frame):  # Ensure snap comes after line set
            # Each frame is 0.1 seconds
            time_diff = (snap_frame - line_set_frame) * 0.1
            # Only include reasonable times (between 1 and 40 seconds)
            if 1 <= time_diff <= 40:
                valid_plays.append((game_id, play_id))
    
    valid_plays_df = pd.DataFrame(valid_plays, columns=['gameId', 'playId'])
    
    # Filter plays data for valid plays only
    filtered_plays = plays_df.merge(valid_plays_df, on=['gameId', 'playId'])
    total_plays = len(filtered_plays)
    
    print(f"\nAnalyzing {total_plays:,} valid plays...")
    
    # Analyze formations and coverages
    off_cross = pd.crosstab(filtered_plays['offenseFormation'], filtered_plays['receiverAlignment'])
    def_cross = pd.crosstab(filtered_plays['pff_passCoverage'], filtered_plays['pff_manZone'])
    
    # Sort formations by total frequency
    formation_totals = off_cross.sum(axis=1)
    off_cross = off_cross.loc[formation_totals.sort_values(ascending=False).index]
    
    # Sort coverages by total frequency and keep top 8
    coverage_totals = def_cross.sum(axis=1)
    top_coverages = coverage_totals.sort_values(ascending=False).head(8)
    def_cross = def_cross.loc[top_coverages.index]
    
    # Set figure style
    plt.rcParams['font.size'] = 14
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Color schemes
    blue_colors = ['#1f77b4', '#3498db', '#5dade2', '#85c1e9', '#aed6f1', '#d6eaf8', '#ebf5fb']
    red_colors = ['#c0392b', '#e74c3c', '#ec7063', '#f1948a', '#f5b7b1', '#fadbd8', '#fdedec']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 18))
    
    # Add title explaining the filtering
    fig.suptitle(f'Formation and Coverage Analysis\nBased on {total_plays:,} plays with valid pre-snap timing (1-40 seconds)',
                 y=0.95, fontsize=16, weight='bold')
    
    # 1. Offensive Formation and Receiver Alignment Chart
    plt.subplot(2, 1, 1)
    
    # Create stacked bar chart
    bottom = np.zeros(len(off_cross))
    bars = []
    min_segment_height = 100
    
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
    
    # Add total count and percentage above each bar
    formation_totals = off_cross.sum(axis=1)  # Calculate total for each formation
    for i, total in enumerate(formation_totals):
        percentage = (total / total_plays) * 100
        plt.text(i, bottom[i] + 50, f'Total: {int(total):,}\n({percentage:.0f}%)',
                ha='center', va='bottom', fontsize=12, weight='bold')
    
    plt.title('Offensive Formation and Receiver Alignment Distribution', pad=20, size=16, weight='bold')
    plt.xlabel('Offensive Formation', size=14, weight='bold')
    plt.ylabel('Number of Plays', size=14, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(title='Receiver Alignment', bbox_to_anchor=(1.05, 1), loc='upper left',
              title_fontsize=14, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 9000)  # Set y-axis limit to 9000 plays
    
    # 2. Defensive Coverage and Man/Zone Chart
    plt.subplot(2, 1, 2)
    
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
    
    # Add total count and percentage above each bar
    coverage_totals = def_cross.sum(axis=1)
    for i, total in enumerate(coverage_totals):
        percentage = (total / total_plays) * 100
        plt.text(i, bottom[i] + 50, f'Total: {total:,}\n({percentage:.0f}%)',
                ha='center', va='bottom', fontsize=12, weight='bold')
    
    plt.title('Defensive Coverage and Man/Zone Distribution', pad=20, size=16, weight='bold')
    plt.xlabel('Coverage Type', size=14, weight='bold')
    plt.ylabel('Number of Plays', size=14, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(title='Coverage Scheme', bbox_to_anchor=(1.05, 1), loc='upper left',
              title_fontsize=14, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 9000)  # Set y-axis limit to 9000 plays
    
    # Adjust layout and save
    plt.tight_layout(pad=4.0, rect=[0, 0, 0.85, 0.95])
    plt.savefig('formation_coverage_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'formation_coverage_analysis.png'")
    
    # Print text summaries
    print("\nOffensive Formation Breakdown:")
    print(formation_totals.to_string())
    
    print("\nDefensive Coverage Breakdown:")
    print(coverage_totals.to_string())

except Exception as e:
    print(f"An error occurred: {str(e)}") 