import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Set style for better visualization
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 6]

# Read all tracking data files
print("Reading tracking data...")
all_tracking_data = []
for week in range(1, 10):  # Weeks 1-9
    df = pd.read_csv(f'tracking_week_{week}.csv')
    all_tracking_data.append(df)

tracking_df = pd.concat(all_tracking_data)

# Group by game and play ID
play_groups = tracking_df.groupby(['gameId', 'playId'])

# Calculate time between line set and snap for each play
snap_times = []
print("\nAnalyzing snap timing...")
for (game_id, play_id), play_data in play_groups:
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
            snap_times.append(time_diff)

# Calculate statistics
avg_time = np.mean(snap_times)
std_time = np.std(snap_times)
median_time = np.median(snap_times)
min_time = np.min(snap_times)
max_time = np.max(snap_times)

# Calculate percentiles
percentiles = np.percentile(snap_times, [25, 75])

print(f"\nAnalysis of {len(snap_times)} valid plays:")
print(f"Average time between line set and snap: {avg_time:.2f} seconds")
print(f"Standard deviation: {std_time:.2f} seconds")
print(f"Median time: {median_time:.2f} seconds")
print(f"Range: {min_time:.2f} to {max_time:.2f} seconds")
print(f"25th percentile: {percentiles[0]:.2f} seconds")
print(f"75th percentile: {percentiles[1]:.2f} seconds")

# Create visualization
fig = plt.figure(figsize=(12, 6))

# Create primary axis for density
ax1 = plt.gca()
ax2 = ax1.twinx()  # Create secondary axis for counts

# Create histogram with density on left axis
n, bins, patches = ax1.hist(snap_times, bins=50, density=True, alpha=0.6, color='skyblue', label='Distribution')

# Calculate actual counts for right axis
counts, _ = np.histogram(snap_times, bins=50)
ax2.hist(snap_times, bins=50, density=False, alpha=0, label='_nolegend_')  # Invisible histogram for count axis

# Add kernel density estimate on left axis
kde = gaussian_kde(snap_times)
x_range = np.linspace(min_time, max_time, 200)
ax1.plot(x_range, kde(x_range), color='navy', linewidth=2, label='Density Estimate')

# Add vertical lines for key statistics
ax1.axvline(avg_time, color='red', linestyle='--', alpha=0.8, label=f'Mean: {avg_time:.2f}s')
ax1.axvline(median_time, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_time:.2f}s')
ax1.axvline(percentiles[0], color='orange', linestyle=':', alpha=0.8, label=f'25th %ile: {percentiles[0]:.2f}s')
ax1.axvline(percentiles[1], color='orange', linestyle=':', alpha=0.8, label=f'75th %ile: {percentiles[1]:.2f}s')

# Customize plot
plt.title('Distribution of Time Between Line Set and Ball Snap\nNFL Weeks 1-9, 2022', fontsize=14, pad=15, weight='bold')
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax2.set_ylabel('Number of Plays', fontsize=12)

# Add filtering information at the bottom
plt.figtext(0.5, -0.05, f'Filtered Data: {len(snap_times):,} plays (excluding times < 1s or > 40s)\nWeeks 1-9 plays with valid line_set and ball_snap events',
            fontsize=14, ha='center')

# Adjust grid
ax1.grid(True, alpha=0.3)

# Adjust legend
ax1.legend(fontsize=10, loc='upper right')

# Save plot
plt.tight_layout()
plt.savefig('snap_timing_distribution.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'snap_timing_distribution.png'") 