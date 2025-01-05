import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_player_entropy_vectorized(player_x, player_y, player_dir, player_speed, all_defenders_x, all_defenders_y, X, Y):
    """Vectorized entropy calculation"""
    # Constants
    w_theta = 0.3
    w_v = 0.2
    v_max = 10.0
    sigma = 2.0
    ball_y = 26.65
    
    # Calculate probability distribution
    p = np.zeros_like(X)
    
    # Use broadcasting for faster computation
    distances = np.sqrt((X[None, :, :] - all_defenders_x[:, None, None])**2 + 
                       (Y[None, :, :] - all_defenders_y[:, None, None])**2)
    p = np.sum(np.exp(-distances**2 / (2*sigma**2)), axis=0)
    
    # Normalize probabilities
    p = p / (np.sum(p) + 1e-10)
    
    # Calculate orientation factor
    theta_relative = np.abs(player_dir - np.degrees(np.arctan2(ball_y - player_y, 60 - player_x)))
    orientation_factor = 1 + w_theta * np.cos(np.radians(theta_relative))
    
    # Calculate velocity factor
    velocity_factor = 1 + w_v * (min(player_speed, v_max) / v_max)
    
    # Calculate entropy
    base_entropy = -np.sum(p * np.log2(p + 1e-10))
    return base_entropy * orientation_factor * velocity_factor

def create_field_grid(grid_size=1.0):
    """Pre-compute the field grid once"""
    x_grid = np.arange(0, 120, grid_size)
    y_grid = np.arange(0, 53.3, grid_size)
    return np.meshgrid(x_grid, y_grid)

def normalize_entropy(entropy_values):
    """Normalize entropy values to 0-100 scale"""
    if len(entropy_values) == 0:
        return []
    min_val = np.min(entropy_values)
    max_val = np.max(entropy_values)
    if max_val == min_val:
        return np.zeros_like(entropy_values)
    return 100 * (entropy_values - min_val) / (max_val - min_val)

def create_heatmap(data, title, output_file):
    """Create and save a heatmap visualization"""
    plt.figure(figsize=(20, 12))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                mask=(data == 0), cbar_kws={'label': 'Normalized Entropy'})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_heatmap(data, position, output_file):
    plt.figure(figsize=(15, 10))
    sns.heatmap(data, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    
    title = f'Formation/Alignment Entropy Differences for {position}\n'
    title += 'Negative (Blue) = More Predictable/Effective Defense\n'
    title += 'Positive (Red) = Less Predictable/Variable Defense'
    
    plt.title(title, pad=20)
    plt.xlabel('Receiver Alignment')
    plt.ylabel('Formation')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def calculate_entropy(velocities):
    # ... existing entropy calculation code ...
    return entropy

def main():
    start_time = time.time()
    print("Loading play and player data...")
    plays_df = pd.read_csv('plays.csv')
    players_df = pd.read_csv('players.csv')[['nflId', 'position']]
    
    # Get unique formations and alignments, handling NaN values
    formations = sorted([f for f in plays_df['offenseFormation'].unique() if pd.notna(f)])
    alignments = sorted([a for a in plays_df['receiverAlignment'].unique() if pd.notna(a)])
    defensive_positions = ['CB', 'SS', 'FS', 'ILB', 'OLB', 'MLB', 'DT', 'DE']
    
    print(f"\nFound {len(formations)} formations and {len(alignments)} alignments")
    print("\nFormations:", formations)
    print("\nAlignments:", alignments)
    
    # Initialize results dictionary
    results = {
        pos: {
            'success': {(f, a): [] for f in formations for a in alignments},
            'failure': {(f, a): [] for f in formations for a in alignments}
        } for pos in defensive_positions
    }
    
    # Pre-compute field grid
    X, Y = create_field_grid()
    
    # Create lookup dictionary for player positions
    player_positions = players_df.set_index('nflId')['position'].to_dict()
    
    # Process all weeks
    total_plays_processed = 0
    all_entropy_values = []  # Store all entropy values for normalization
    
    for week in range(1, 10):
        print(f"\nProcessing Week {week}...")
        tracking_df = pd.read_csv(f'tracking_week_{week}.csv',
                                usecols=['gameId', 'playId', 'nflId', 'frameId', 'event',
                                        'x', 'y', 'dir', 's', 'club'])
        print(f"Loaded tracking data: {len(tracking_df):,} rows")
        
        # Process plays
        print("\nAnalyzing plays...")
        for (game_id, play_id), play_data in tqdm(tracking_df.groupby(['gameId', 'playId'])):
            # Get snap frames
            events = play_data.sort_values('frameId')
            line_set_frame = events[events['event'] == 'line_set']['frameId'].min()
            snap_frame = events[events['event'] == 'ball_snap']['frameId'].min()
            
            if not (pd.notna(line_set_frame) and pd.notna(snap_frame) and 
                    snap_frame > line_set_frame):
                continue
                
            time_diff = (snap_frame - line_set_frame) * 0.1
            if not (1 <= time_diff <= 40):
                continue
            
            # Get play details
            play_details = plays_df[
                (plays_df['gameId'] == game_id) & 
                (plays_df['playId'] == play_id)
            ].iloc[0]
            
            formation = play_details['offenseFormation']
            alignment = play_details['receiverAlignment']
            
            # Skip plays with missing formation or alignment
            if pd.isna(formation) or pd.isna(alignment):
                continue
                
            defensive_team = play_details['defensiveTeam']
            
            # Filter for relevant frames and defensive team
            play_frames = play_data[
                (play_data['frameId'] >= line_set_frame) & 
                (play_data['frameId'] <= snap_frame) &
                (play_data['club'] == defensive_team)
            ]
            
            if play_frames.empty:
                continue
            
            # Determine if defense was successful
            yards = play_details['yardsGained']
            is_no_gain = yards <= 0
            is_pass_defense = play_details['passResult'] == 'I'
            is_qb_hit = not pd.isna(play_details['timeToSack'])
            is_success = any([is_no_gain, is_pass_defense, is_qb_hit])
            category = 'success' if is_success else 'failure'
            
            # Process each defensive player
            for _, frame in play_frames.iterrows():
                player_id = frame['nflId']
                position = player_positions.get(player_id)
                
                if position not in defensive_positions:
                    continue
                
                # Get all defenders' positions for this frame
                defenders_same_frame = play_frames[play_frames['frameId'] == frame['frameId']]
                
                # Calculate entropy
                entropy = calculate_player_entropy_vectorized(
                    frame['x'], frame['y'], frame['dir'], frame['s'],
                    defenders_same_frame['x'].values,
                    defenders_same_frame['y'].values,
                    X, Y
                )
                
                results[position][category][(formation, alignment)].append(entropy)
                all_entropy_values.append(entropy)
            
            total_plays_processed += 1
            if total_plays_processed % 100 == 0:
                elapsed_time = time.time() - start_time
                plays_per_second = total_plays_processed / elapsed_time
                print(f"\nProcessed {total_plays_processed:,} total plays")
                print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
                print(f"Processing speed: {plays_per_second:.2f} plays/second")
    
    # Create results tables and statistical analysis
    print("\nGenerating results...")
    
    for position in defensive_positions:
        # Create DataFrames for success and failure
        success_data = pd.DataFrame(index=formations, columns=alignments, dtype=float)
        failure_data = pd.DataFrame(index=formations, columns=alignments, dtype=float)
        p_values = pd.DataFrame(index=formations, columns=alignments, dtype=float)
        
        # Calculate averages and p-values
        for formation in formations:
            for alignment in alignments:
                success_values = results[position]['success'][(formation, alignment)]
                failure_values = results[position]['failure'][(formation, alignment)]
                
                if len(success_values) > 0 and len(failure_values) > 0:
                    # Normalize values
                    success_norm = normalize_entropy(success_values)
                    failure_norm = normalize_entropy(failure_values)
                    
                    success_data.loc[formation, alignment] = np.mean(success_norm)
                    failure_data.loc[formation, alignment] = np.mean(failure_norm)
                    
                    # Calculate p-value
                    _, p_value = stats.ttest_ind(success_norm, failure_norm)
                    p_values.loc[formation, alignment] = p_value
                else:
                    success_data.loc[formation, alignment] = 0
                    failure_data.loc[formation, alignment] = 0
                    p_values.loc[formation, alignment] = 1
        
        # Create difference heatmap
        diff_data = success_data - failure_data
        
        # Save results
        success_data.to_csv(f'formation_entropy_{position}_success.csv')
        failure_data.to_csv(f'formation_entropy_{position}_failure.csv')
        diff_data.to_csv(f'formation_entropy_{position}_diff.csv')
        
        # Create heatmaps
        create_heatmap(diff_data, 
                      f'Entropy Difference by Formation/Alignment - {position}',
                      f'formation_entropy_{position}_heatmap.png')
        
        # Print summary
        print(f"\nResults for {position}:")
        print("\nTop 5 Formation/Alignment Combinations with Significant Differences (p < 0.05):")
        significant_diffs = []
        for formation in formations:
            for alignment in alignments:
                if p_values.loc[formation, alignment] < 0.05:
                    diff = diff_data.loc[formation, alignment]
                    significant_diffs.append((formation, alignment, diff, p_values.loc[formation, alignment]))
        
        significant_diffs.sort(key=lambda x: abs(x[2]), reverse=True)
        for formation, alignment, diff, p_value in significant_diffs[:5]:
            print(f"{formation} - {alignment}: {diff:.2f} (p={p_value:.3e})")
    
    print(f"\nTotal analysis time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Total plays analyzed: {total_plays_processed:,}")

if __name__ == "__main__":
    main() 