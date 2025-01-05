import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(data, position, ax):
    # Create mask for zero values
    mask = (data == 0)
    
    sns.heatmap(data, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
                mask=mask, cbar_kws={'label': 'Entropy Difference'}, ax=ax,
                annot_kws={'size': 10})
    
    # Position title and explanation with better spacing
    ax.text(0.5, 1.25, position, fontsize=20, fontweight='bold',
            horizontalalignment='center', transform=ax.transAxes)
    
    ax.text(0.5, 1.15, 'Negative (Blue) = More Predictable/Effective Defense\n'
            'Positive (Red) = Less Predictable/Variable Defense',
            horizontalalignment='center', transform=ax.transAxes,
            fontsize=12)
    
    ax.set_xlabel('Receiver Alignment', fontsize=12)
    ax.set_ylabel('Formation', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

def main():
    positions = ['CB', 'SS', 'FS', 'ILB', 'OLB', 'MLB', 'DT', 'DE']
    
    # Create a 4x2 subplot figure with extra space at top for main title
    fig = plt.figure(figsize=(24, 34))
    
    # Add main title with more space above plots
    fig.suptitle('Defensive Position Entropy Analysis by Formation and Alignment', 
                 fontsize=28, y=0.98, fontweight='bold')
    
    # Create subplot grid with appropriate spacing
    gs = fig.add_gridspec(4, 2, hspace=0.6, wspace=0.2)  # Increased hspace for title
    axes = []
    for i in range(4):
        for j in range(2):
            axes.append(fig.add_subplot(gs[i, j]))
    
    for idx, position in enumerate(positions):
        print(f"Processing {position}...")
        
        # Read the difference CSV file
        diff_df = pd.read_csv(f'formation_entropy_{position}_diff.csv', index_col=0)
        
        # Create heatmap in the corresponding subplot
        plot_heatmap(diff_df, position, axes[idx])
    
    plt.savefig('combined_entropy_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created combined_entropy_heatmaps.png")

if __name__ == "__main__":
    main() 