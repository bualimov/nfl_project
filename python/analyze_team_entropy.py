import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
from scipy import stats

def create_field_grid(grid_size=1.0):
    """Pre-compute the field grid once"""
    x_grid = np.arange(0, 120, grid_size)
    y_grid = np.arange(0, 53.3, grid_size)
    return np.meshgrid(x_grid, y_grid)

def calculate_entropy(frame_data, play_direction):
    """Calculate entropy for a set of defensive players."""
    if len(frame_data) == 0:
        return 0
    
    # Constants
    sigma = 1.0  # Spatial spread parameter
    w_theta = 0.3  # Weight for orientation
    w_v = 0.2  # Weight for velocity
    v_max = 10.0  # Maximum velocity threshold
    ball_y = 26.65  # Middle of the field width (53.3/2)
    grid_size = 1.0  # Grid size in yards
    
    # Create grid for the entire field
    x_grid = np.arange(0, 120, grid_size)
    y_grid = np.arange(0, 53.3, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calculate probability distribution
    p = np.zeros_like(X)
    
    # Get player positions
    positions = frame_data[['x', 'y']].values
    directions = frame_data['dir'].values
    speeds = frame_data['s'].values
    
    # Calculate spatial distribution
    for pos in positions:
        dist = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2)
        p += np.exp(-dist**2 / (2*sigma**2))
    
    # Normalize probabilities
    p = p / (np.sum(p) + 1e-10)
    
    # Calculate base entropy
    base_entropy = -np.sum(p * np.log2(p + 1e-10))
    
    # Calculate orientation factor
    target_x = 120 if play_direction == 'right' else 0
    orientation_factors = []
    for i, (pos, direction) in enumerate(zip(positions, directions)):
        theta_relative = np.abs(direction - np.degrees(np.arctan2(ball_y - pos[1], target_x - pos[0])))
        orientation_factor = 1 + w_theta * np.cos(np.radians(theta_relative))
        orientation_factors.append(orientation_factor)
    
    # Calculate velocity factor
    velocity_factors = []
    for speed in speeds:
        velocity_factor = 1 + w_v * (min(speed, v_max) / v_max)
        velocity_factors.append(velocity_factor)
    
    # Combine all factors
    total_entropy = base_entropy * np.mean(orientation_factors) * np.mean(velocity_factors)
    
    return total_entropy

def normalize_entropy(entropy_value, min_entropy, max_entropy):
    """Normalize entropy value to range [0,1]."""
    if max_entropy == min_entropy:
        return 0
    return (entropy_value - min_entropy) / (max_entropy - min_entropy)

def calculate_team_entropy():
    # Load plays data
    plays = pd.read_csv('plays.csv')
    print("\nLoaded plays data")
    
    # Get games data for home/away team info
    games = pd.read_csv('games.csv')
    print("Loaded games data")
    
    # Merge plays with games to get home/away team info
    plays = plays.merge(games[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']], on='gameId', how='left')
    
    # Add defensiveTeam column
    plays['defensiveTeam'] = plays.apply(
        lambda x: x['visitorTeamAbbr'] if x['possessionTeam'] == x['homeTeamAbbr'] else x['homeTeamAbbr'],
        axis=1
    )
    
    # First pass: identify valid plays based on timing criteria
    valid_plays_list = []
    print("\nProcessing tracking data to identify valid plays...")
    
    # Process tracking data to apply timing criteria
    for week in range(1, 10):
        filename = f'tracking_week_{week}.csv'
        if not os.path.exists(filename):
            continue
            
        print(f"\nLoading {filename}")
        tracking_data = pd.read_csv(filename)
        
        # Group by game and play ID
        play_groups = tracking_data.groupby(['gameId', 'playId'])
        
        # Process each play
        for (game_id, play_id), play_data in play_groups:
            # Get events in chronological order
            events = play_data.sort_values('frameId')
            
            # Find line_set and ball_snap events
            line_set_frame = events[events['event'] == 'line_set']['frameId'].min()
            snap_frame = events[events['event'] == 'ball_snap']['frameId'].min()
            
            # If both events exist, calculate time difference
            if pd.notna(line_set_frame) and pd.notna(snap_frame):
                # Calculate time difference (10 fps)
                time_diff = (snap_frame - line_set_frame) * 0.1
                
                # Apply timing criteria
                if time_diff > 0 and 1 <= time_diff <= 40:
                    valid_plays_list.append((game_id, play_id))
    
    # Convert to DataFrame for easier filtering
    valid_plays_df = pd.DataFrame(valid_plays_list, columns=['gameId', 'playId'])
    
    # Get final set of valid plays
    valid_plays = plays.merge(valid_plays_df, on=['gameId', 'playId'], how='inner')
    print(f"\nFiltered to {len(valid_plays)} valid plays after timing criteria")
    
    # Progress bar setup for entropy calculation
    total_plays = len(valid_plays)
    start_time = time.time()
    pbar = tqdm(total=total_plays, desc='Analyzing plays')
    plays_processed = 0
    
    # Initialize team entropy dictionary
    team_entropy_dict = {}
    
    # Process each week
    for week in range(1, 10):
        filename = f'tracking_week_{week}.csv'
        if not os.path.exists(filename):
            continue
            
        print(f"\nLoading {filename}")
        tracking_data = pd.read_csv(filename)
        
        # Get plays for this week
        week_plays = valid_plays[valid_plays['gameId'].isin(tracking_data['gameId'].unique())]
        
        # Process each play
        for _, play in week_plays.iterrows():
            game_id = play['gameId']
            play_id = play['playId']
            
            # Get play data
            play_data = tracking_data[
                (tracking_data['gameId'] == game_id) & 
                (tracking_data['playId'] == play_id)
            ]
            
            # Get frames between line set and snap
            line_set_frame = play_data[play_data['event'] == 'line_set']['frameId'].min()
            snap_frame = play_data[play_data['event'] == 'ball_snap']['frameId'].min()
            
            # Get play direction
            play_direction = play_data['playDirection'].iloc[0]
            
            # Get relevant frames
            play_frames = play_data[
                (play_data['frameId'] >= line_set_frame) & 
                (play_data['frameId'] <= snap_frame)
            ]
            
            # Process each frame
            frame_groups = play_frames.groupby('frameId')
            frame_entropies = []
            
            for _, frame in frame_groups:
                # Get defensive players
                defensive_players = frame[frame['club'] == play['defensiveTeam']]
                if len(defensive_players) == 0:
                    continue
                
                # Calculate entropy for this frame
                frame_entropy = calculate_entropy(defensive_players, play_direction)
                frame_entropies.append(frame_entropy)
            
            # Calculate average entropy for the play
            if frame_entropies:
                play_entropy = np.mean(frame_entropies)
                
                # Store entropy value
                defensive_team = play['defensiveTeam']
                if defensive_team not in team_entropy_dict:
                    team_entropy_dict[defensive_team] = []
                team_entropy_dict[defensive_team].append(play_entropy)
            
            # Update progress
            plays_processed += 1
            pbar.update(1)
            
            # Print progress every 1000 plays
            if plays_processed % 1000 == 0:
                elapsed_time = time.time() - start_time
                plays_per_second = plays_processed / elapsed_time
                print(f"\nProcessed {plays_processed}/{total_plays} plays")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                print(f"Processing speed: {plays_per_second:.2f} plays/second")
    
    pbar.close()
    
    # Calculate statistics for each team
    team_stats = []
    all_entropy_values = []
    for team, entropy_values in team_entropy_dict.items():
        entropy_values = np.array(entropy_values)
        all_entropy_values.extend(entropy_values)
        
        stats_dict = {
            'team': team,
            'mean_entropy': np.mean(entropy_values),
            'median_entropy': np.median(entropy_values),
            'q1': np.percentile(entropy_values, 25),
            'q3': np.percentile(entropy_values, 75),
            'min': np.min(entropy_values),
            'max': np.max(entropy_values),
            'std_dev': np.std(entropy_values),
            'sample_size': len(entropy_values)
        }
        team_stats.append(stats_dict)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(team_stats)
    
    # Calculate normalized entropy values
    min_entropy = results_df['mean_entropy'].min()
    max_entropy = results_df['mean_entropy'].max()
    results_df['normalized_entropy'] = results_df['mean_entropy'].apply(
        lambda x: normalize_entropy(x, min_entropy, max_entropy)
    )
    
    # Sort by mean entropy
    results_df = results_df.sort_values('mean_entropy')
    
    # Save results
    results_df.to_csv('team_entropy_analysis.csv', index=False)
    print("\nAnalysis complete. Results saved to team_entropy_analysis.csv")
    print(f"\nTotal plays analyzed: {plays_processed}")
    print(f"Total teams analyzed: {len(team_stats)}")

if __name__ == "__main__":
    calculate_team_entropy() 