import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os

def calculate_player_entropy(player_frames):
    """Calculate entropy for a player's frames"""
    # Normalize position data
    x_pos = player_frames['x']
    y_pos = player_frames['y']
    orientation = player_frames['dir']
    speed = player_frames['s']
    
    # Calculate position-based entropy
    pos_entropy = np.std(x_pos) * np.std(y_pos)
    
    # Calculate orientation factor (normalized to [0,1])
    orientation_std = np.std(orientation) / 180.0
    
    # Calculate speed factor (normalized to [0,1])
    speed_factor = np.mean(speed) / 10.0  # Assuming max speed ~10 yards/s
    
    # Combine factors with weights
    w_pos = 1.0
    w_orient = 0.3
    w_speed = 0.2
    
    total_entropy = (w_pos * pos_entropy + 
                    w_orient * orientation_std + 
                    w_speed * speed_factor)
    
    # Normalize to 0-100 scale (adjusted range based on typical values)
    normalized_entropy = (total_entropy - 100) / (200 - 100) * 100
    normalized_entropy = np.clip(normalized_entropy, 0, 100)  # Ensure values stay within 0-100
    
    return total_entropy, normalized_entropy

def main():
    print("Starting MLB entropy analysis...")
    start_time = time.time()
    
    # Load tracking data for all weeks
    all_weeks_data = []
    for week in range(1, 10):
        week_data = pd.read_csv(f'tracking_week_{week}.csv')
        all_weeks_data.append(week_data)
    
    tracking_data = pd.concat(all_weeks_data, ignore_index=True)
    print(f"Loaded tracking data for {len(all_weeks_data)} weeks")
    
    # Load players data
    players_data = pd.read_csv('players.csv')
    mlb_players = players_data[players_data['position'] == 'MLB']
    
    # Initialize results dictionary
    results = {
        'nflId': [],
        'displayName': [],
        'actual_entropy': [],
        'normalized_entropy': [],
        'total_frames': []
    }
    
    # Process each MLB player
    total_players = len(mlb_players)
    print(f"\nAnalyzing {total_players} MLB players...")
    
    for idx, player in tqdm(mlb_players.iterrows(), total=total_players):
        player_frames = tracking_data[tracking_data['nflId'] == player['nflId']]
        
        if len(player_frames) > 0:
            actual_entropy, normalized_entropy = calculate_player_entropy(player_frames)
            
            results['nflId'].append(player['nflId'])
            results['displayName'].append(player['displayName'])
            results['actual_entropy'].append(actual_entropy)
            results['normalized_entropy'].append(normalized_entropy)
            results['total_frames'].append(len(player_frames))
    
    # Create DataFrame and sort by entropy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('normalized_entropy', ascending=False)
    
    # Save results
    results_df.to_csv('mlb_player_entropy_analysis.csv', index=False)
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis complete! Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Results saved to mlb_player_entropy_analysis.csv")
    print(f"\nProcessed {len(results_df)} MLB players")
    print(f"Average normalized entropy: {results_df['normalized_entropy'].mean():.2f}")
    print(f"Average actual entropy: {results_df['actual_entropy'].mean():.2f}")

if __name__ == "__main__":
    main() 