import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the results
results_df = pd.read_csv('defensive_entropy_analysis_all_weeks.csv')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_position_comparison_plot():
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Prepare data
    positions = results_df['Position']
    success_avg = results_df['Success_Avg_Entropy']
    failure_avg = results_df['Failure_Avg_Entropy']
    
    # Set bar width
    bar_width = 0.35
    r1 = np.arange(len(positions))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    plt.bar(r1, success_avg, width=bar_width, label='Successful Plays', color='green', alpha=0.7)
    plt.bar(r2, failure_avg, width=bar_width, label='Failed Plays', color='red', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Defensive Position')
    plt.ylabel('Normalized Entropy (0-100)')
    plt.title('Average Entropy by Position and Play Outcome')
    plt.xticks([r + bar_width/2 for r in range(len(positions))], positions)
    
    # Add legend
    plt.legend()
    
    # Add significance stars
    for i in range(len(positions)):
        p_value = results_df['P_Value'].iloc[i]
        if p_value < 0.001:
            plt.text(i + bar_width/2, max(success_avg[i], failure_avg[i]) + 1, '***', 
                    ha='center', va='bottom')
    
    # Save plot
    plt.tight_layout()
    plt.savefig('entropy_by_position.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sample_size_plot():
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    positions = results_df['Position']
    total_samples = results_df['Total_Plays']
    
    # Create bar plot
    bars = plt.bar(positions, total_samples)
    
    # Add labels and title
    plt.xlabel('Defensive Position')
    plt.ylabel('Number of Player-Frames')
    plt.title('Sample Size by Position')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', rotation=45)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('sample_size_by_position.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_entropy_difference_plot():
    # Create figure with white background
    plt.figure(figsize=(12, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Add black border
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    # Prepare data
    positions = results_df['Position']
    differences = results_df['Entropy_Difference']
    
    # Create bar plot with color based on positive/negative
    colors = ['green' if x >= 0 else 'red' for x in differences]
    bars = plt.bar(positions, differences, color=colors, alpha=0.7)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title with increased size and bold weight
    plt.xlabel('Defensive Position', size=20, weight='bold', labelpad=10)
    plt.ylabel('Entropy Difference (Success - Failure)', size=20, weight='bold', labelpad=10)
    plt.title('Entropy Difference by Position', size=20, weight='bold', pad=15)
    
    # Set tick label font sizes and remove gridlines
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(False)
    
    # Set y-axis limits
    plt.ylim(-4, 2)
    
    # Add value labels on top/bottom of bars with increased spacing
    for bar in bars:
        height = bar.get_height()
        y_offset = 0.2 if height >= 0 else -0.2  # Increased offset for label spacing
        plt.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                f'{height:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=14)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('entropy_difference_by_position.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_success_rate_plot():
    # Calculate success rate for each position
    results_df['Success_Rate'] = (results_df['Success_Count'] / 
                                 (results_df['Success_Count'] + results_df['Failure_Count']) * 100)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    bars = plt.bar(results_df['Position'], results_df['Success_Rate'])
    
    # Add labels and title
    plt.xlabel('Defensive Position')
    plt.ylabel('Success Rate (%)')
    plt.title('Defensive Success Rate by Position')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Save plot
    plt.tight_layout()
    plt.savefig('success_rate_by_position.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create all visualizations
    print("Creating position comparison plot...")
    create_position_comparison_plot()
    
    print("Creating sample size plot...")
    create_sample_size_plot()
    
    print("Creating entropy difference plot...")
    create_entropy_difference_plot()
    
    print("Creating success rate plot...")
    create_success_rate_plot()
    
    print("\nAll visualizations have been created!")

if __name__ == "__main__":
    main() 