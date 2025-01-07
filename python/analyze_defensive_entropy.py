import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from scipy import stats

def create_field_grid(grid_size=1.0):
    """Pre-compute the field grid once"""
    x_grid = np.arange(0, 120, grid_size)
    y_grid = np.arange(0, 53.3, grid_size)
    return np.meshgrid(x_grid, y_grid)

def calculate_player_entropy_vectorized(player_x, player_y, player_dir, player_speed,
                                all_defenders_x, all_defenders_y, X, Y, play_direction):
    # Constants
    sigma = 1.0  # Spatial spread parameter
    w_theta = 0.3  # Weight for orientation
    w_v = 0.2  # Weight for velocity
    v_max = 10.0  # Maximum velocity threshold
    ball_y = 26.65  # Middle of the field width (53.3/2)

    # Use broadcasting for faster computation
    distances = np.sqrt((X[None, :, :] - all_defenders_x[:, None, None])**2 + 
                       (Y[None, :, :] - all_defenders_y[:, None, None])**2)
    p = np.sum(np.exp(-distances**2 / (2*sigma**2)), axis=0)
    
    # Normalize probabilities
    p = p / (np.sum(p) + 1e-10)
    
    # Calculate orientation factor based on play direction
    target_x = 120 if play_direction == 'right' else 0
    theta_relative = np.abs(player_dir - np.degrees(np.arctan2(ball_y - player_y, target_x - player_x)))
    orientation_factor = 1 + w_theta * np.cos(np.radians(theta_relative))
    
    # Calculate velocity factor
    velocity_factor = 1 + w_v * (min(player_speed, v_max) / v_max)
    
    # Calculate entropy
    base_entropy = -np.sum(p * np.log2(p + 1e-10))
    return base_entropy * orientation_factor * velocity_factor

def normalize_entropy(entropy_values):
    """Normalize entropy values to 0-100 scale"""
    min_val = np.min(entropy_values)
    max_val = np.max(entropy_values)
    return 100 * (entropy_values - min_val) / (max_val - min_val)

def main():
    start_time = time.time()
    print("Loading play and player data...")
    plays_df = pd.read_csv('plays.csv')
    players_df = pd.read_csv('players.csv')[['nflId', 'position']]
    
    # Initialize results dictionary with sets for faster lookups
    all_positions = ['CB', 'SS', 'FS', 'ILB', 'OLB', 'MLB', 'DT', 'DE']
    position_results = {
        'success': {pos: [] for pos in all_positions},
        'failure': {pos: [] for pos in all_positions}
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
                                        'x', 'y', 'dir', 's', 'club', 'playDirection'])
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
            
            # Get play direction from tracking data
            play_direction = play_frames['playDirection'].iloc[0]
            
            # Process each defensive player
            for _, frame in play_frames.iterrows():
                player_id = frame['nflId']
                position = player_positions.get(player_id)
                
                if position not in position_results[category]:
                    continue
                
                # Get all defenders' positions for this frame
                defenders_same_frame = play_frames[play_frames['frameId'] == frame['frameId']]
                
                # Calculate entropy
                entropy = calculate_player_entropy_vectorized(
                    frame['x'], frame['y'], frame['dir'], frame['s'],
                    defenders_same_frame['x'].values,
                    defenders_same_frame['y'].values,
                    X, Y, play_direction
                )
                
                position_results[category][position].append(entropy)
                all_entropy_values.append(entropy)
            
            total_plays_processed += 1
            if total_plays_processed % 100 == 0:
                elapsed_time = time.time() - start_time
                plays_per_second = total_plays_processed / elapsed_time
                print(f"\nProcessed {total_plays_processed:,} total plays")
                print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
                print(f"Processing speed: {plays_per_second:.2f} plays/second")
    
    # Normalize all entropy values
    all_entropy_values = np.array(all_entropy_values)
    normalized_values = normalize_entropy(all_entropy_values)
    
    # Calculate normalized averages and perform statistical analysis
    print("\nAnalysis Results:")
    print("\nDefensive Position Entropy Analysis (All Weeks)")
    print("=" * 80)
    print(f"{'Position':<8} {'Success Avg':<12} {'Failure Avg':<12} {'Difference':<12} {'Sample Size':<12} {'P-Value':<12}")
    print("-" * 80)
    
    results_data = []
    all_success_entropy = []
    all_failure_entropy = []
    
    for position in sorted(position_results['success'].keys()):
        success_values = np.array(position_results['success'][position])
        failure_values = np.array(position_results['failure'][position])
        
        # Normalize values
        success_norm = normalize_entropy(success_values)
        failure_norm = normalize_entropy(failure_values)
        
        # Calculate statistics
        success_avg = np.mean(success_norm)
        failure_avg = np.mean(failure_norm)
        success_count = len(success_values)
        failure_count = len(failure_values)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(success_norm, failure_norm)
        
        print(f"{position:<8} {success_avg:>11.2f} {failure_avg:>11.2f} "
              f"{(success_avg - failure_avg):>11.2f} {success_count + failure_count:>11} {p_value:>11.3e}")
        
        # Accumulate for overall defensive entropy
        all_success_entropy.extend(success_norm)
        all_failure_entropy.extend(failure_norm)
        
        results_data.append({
            'Position': position,
            'Success_Avg_Entropy': success_avg,
            'Failure_Avg_Entropy': failure_avg,
            'Entropy_Difference': success_avg - failure_avg,
            'Success_Count': success_count,
            'Failure_Count': failure_count,
            'Total_Plays': success_count + failure_count,
            'P_Value': p_value
        })
    
    # Calculate overall defensive entropy
    overall_success_avg = np.mean(all_success_entropy)
    overall_failure_avg = np.mean(all_failure_entropy)
    overall_diff = overall_success_avg - overall_failure_avg
    t_stat, p_value = stats.ttest_ind(all_success_entropy, all_failure_entropy)
    
    print("\nOverall Defensive Entropy:")
    print(f"Success Average: {overall_success_avg:.2f}")
    print(f"Failure Average: {overall_failure_avg:.2f}")
    print(f"Difference: {overall_diff:.2f}")
    print(f"P-Value: {p_value:.3e}")
    
    # Save results
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('defensive_entropy_analysis_all_weeks.csv', index=False)
    print("\nDetailed results saved to 'defensive_entropy_analysis_all_weeks.csv'")
    
    total_time = time.time() - start_time
    print(f"\nTotal analysis time: {total_time/60:.1f} minutes")
    print(f"Total plays analyzed: {total_plays_processed:,}")

if __name__ == "__main__":
    main() 