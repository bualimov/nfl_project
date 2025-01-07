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
    """Vectorized entropy calculation"""
    # Constants - Same as DT analysis for consistency
    sigma = 1.0  # Spatial spread parameter
    w_theta = 0.3  # Weight for orientation
    w_v = 0.2  # Weight for velocity
    v_max = 10.0  # Maximum velocity threshold
    ball_y = 26.65  # Middle of the field width (53.3/2)
    
    # Calculate probability distribution
    p = np.zeros_like(X)
    
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
    if len(entropy_values) == 0:
        return []
    min_val = np.min(entropy_values)
    max_val = np.max(entropy_values)
    if max_val == min_val:
        return np.zeros_like(entropy_values)
    return 100 * (entropy_values - min_val) / (max_val - min_val)

def main():
    start_time = time.time()
    print("Loading play and player data...")
    plays_df = pd.read_csv('plays.csv')
    players_df = pd.read_csv('players.csv')
    
    # Filter for OLB players
    olb_players = players_df[players_df['position'] == 'OLB']
    print(f"\nFound {len(olb_players)} OLB players")
    
    # Get jersey numbers from first week's tracking data
    print("Loading Week 1 data to get jersey numbers...")
    week1_df = pd.read_csv('tracking_week_1.csv')
    
    # Create player info lookup with jersey numbers
    olb_info = {}
    for nfl_id in olb_players['nflId']:
        player_data = week1_df[week1_df['nflId'] == nfl_id]
        if not player_data.empty:
            jersey_num = player_data['jerseyNumber'].iloc[0]
            display_name = olb_players[olb_players['nflId'] == nfl_id]['displayName'].iloc[0]
            olb_info[nfl_id] = {
                'name': f"{display_name} (#{int(jersey_num)})",
                'entropy_values': [],
                'frame_count': 0,
                'play_count': 0,
                'avg_distance_to_los': [],
                'avg_speed': []
            }
    
    print(f"Found jersey numbers for {len(olb_info)} OLB players")
    
    # Pre-compute field grid
    X, Y = create_field_grid()
    
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
            
            # Get play direction from tracking data
            play_direction = play_frames['playDirection'].iloc[0]
            
            # Get line of scrimmage
            los_yard = play_details['absoluteYardlineNumber']
            
            # Process each OLB player
            for player_id in olb_info.keys():
                player_frames = play_frames[play_frames['nflId'] == player_id]
                if not player_frames.empty:
                    olb_info[player_id]['play_count'] += 1
                    
                    # Calculate average distance to line of scrimmage
                    avg_dist_to_los = np.mean(np.abs(player_frames['x'] - los_yard))
                    olb_info[player_id]['avg_distance_to_los'].append(avg_dist_to_los)
                    
                    # Calculate average speed
                    avg_speed = np.mean(player_frames['s'])
                    olb_info[player_id]['avg_speed'].append(avg_speed)
                    
                    for _, frame in player_frames.iterrows():
                        # Get all defenders' positions for this frame
                        defenders_same_frame = play_frames[play_frames['frameId'] == frame['frameId']]
                        
                        # Calculate entropy
                        entropy = calculate_player_entropy_vectorized(
                            frame['x'], frame['y'], frame['dir'], frame['s'],
                            defenders_same_frame['x'].values,
                            defenders_same_frame['y'].values,
                            X, Y, play_direction
                        )
                        
                        olb_info[player_id]['entropy_values'].append(entropy)
                        olb_info[player_id]['frame_count'] += 1
                        all_entropy_values.append(entropy)
            
            total_plays_processed += 1
            if total_plays_processed % 100 == 0:
                elapsed_time = time.time() - start_time
                plays_per_second = total_plays_processed / elapsed_time
                print(f"\nProcessed {total_plays_processed:,} total plays")
                print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
                print(f"Processing speed: {plays_per_second:.2f} plays/second")
    
    # Calculate final results
    print("\nCalculating final results...")
    results_data = []
    
    # Normalize all entropy values
    all_entropy_values = np.array(all_entropy_values)
    normalized_values = normalize_entropy(all_entropy_values)
    
    for player_id, data in olb_info.items():
        if data['frame_count'] > 0:  # Only include players with data
            raw_entropy = np.mean(data['entropy_values'])
            norm_entropy = np.mean(normalize_entropy(data['entropy_values']))
            avg_dist_to_los = np.mean(data['avg_distance_to_los']) if data['avg_distance_to_los'] else 0
            avg_speed = np.mean(data['avg_speed']) if data['avg_speed'] else 0
            
            results_data.append({
                'Player': data['name'],
                'Raw_Entropy': raw_entropy,
                'Normalized_Entropy': norm_entropy,
                'Frame_Count': data['frame_count'],
                'Play_Count': data['play_count'],
                'Avg_Distance_to_LOS': avg_dist_to_los,
                'Avg_Speed': avg_speed
            })
    
    # Create DataFrame and sort by normalized entropy
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('Normalized_Entropy', ascending=False)
    
    # Save results
    results_df.to_csv('olb_player_entropy_analysis.csv', index=False)
    print("\nResults saved to 'olb_player_entropy_analysis.csv'")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total plays analyzed: {total_plays_processed:,}")
    print(f"Total OLBs analyzed: {len(results_df)}")
    print(f"Average normalized entropy: {results_df['Normalized_Entropy'].mean():.2f}")
    print(f"Average distance to LOS: {results_df['Avg_Distance_to_LOS'].mean():.2f} yards")
    print(f"Average speed: {results_df['Avg_Speed'].mean():.2f} yards/second")
    print(f"Analysis time: {(time.time() - start_time)/60:.1f} minutes")

if __name__ == "__main__":
    main() 