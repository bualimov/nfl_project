import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(data, position, ax, vmin, vmax, add_colorbar=False):
    # Create mask for zero values
    mask = (data == 0)
    
    # Create heatmap without colorbar
    sns.heatmap(data, annot=True, cmap='RdBu_r', center=0, fmt='.1f',
                mask=mask, ax=ax, cbar=add_colorbar,
                annot_kws={'size': 18}, vmin=vmin, vmax=vmax)
    
    # Add light border around the heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('gray')
        spine.set_linewidth(1.0)
    
    # Position title without explanation (moved to main title)
    ax.text(0.5, 1.08, position, fontsize=30, fontweight='bold',
            horizontalalignment='center', transform=ax.transAxes)
    
    ax.set_xlabel('Receiver Alignment', fontsize=24, fontweight='bold', labelpad=15)
    ax.set_ylabel('Formation', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18)

def main():
    positions = ['CB', 'SS', 'FS', 'ILB', 'OLB', 'MLB', 'DT', 'DE']
    
    # First pass: find global min and max values
    all_values = []
    for position in positions:
        diff_df = pd.read_csv(f'formation_entropy_{position}_diff.csv', index_col=0)
        valid_values = diff_df[(diff_df != 0) & (~diff_df.isna())].values.flatten()
        all_values.extend(valid_values)
    
    all_values = np.array(all_values)
    vmin = np.nanmin(all_values)
    vmax = np.nanmax(all_values)
    
    # Make the scale symmetric around zero
    abs_max = max(abs(vmin), abs(vmax))
    vmin = -abs_max
    vmax = abs_max
    
    print(f"Using symmetric range: {vmin:.2f} to {vmax:.2f}")
    
    # Create figure with extra space at top for colorbar
    fig = plt.figure(figsize=(24, 38))  # Increased height for extra title space
    
    # Add main three-line title
    fig.text(0.5, 0.98, 'Defensive Position Entropy Analysis by Formation and Alignment',
             fontsize=30, weight='bold', ha='center')
    fig.text(0.5, 0.965, 'Negative (Blue) = More Predictable/Effective Defense',
             fontsize=28, ha='center')
    fig.text(0.5, 0.955, 'Positive (Red) = Less Predictable/Variable Defense',
             fontsize=28, ha='center')
    
    # Create a special axes for the colorbar at the top
    cbar_ax = fig.add_axes([0.15, 0.92, 0.7, 0.02])  # Moved colorbar down
    
    # Create the colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Entropy Difference')
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('Entropy Difference', size=24, weight='bold', y=-0.8)
    
    # Create subplot grid with appropriate spacing
    gs = fig.add_gridspec(4, 2, hspace=0.6, wspace=0.4, top=0.85)  # Increased wspace from 0.2 to 0.4
    axes = []
    for i in range(4):
        for j in range(2):
            axes.append(fig.add_subplot(gs[i, j]))
    
    for idx, position in enumerate(positions):
        print(f"Processing {position}...")
        diff_df = pd.read_csv(f'formation_entropy_{position}_diff.csv', index_col=0)
        plot_heatmap(diff_df, position, axes[idx], vmin, vmax, add_colorbar=False)
    
    plt.savefig('combined_entropy_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created combined_entropy_heatmaps.png")
    print(f"Global value range: {vmin:.2f} to {vmax:.2f}")

if __name__ == "__main__":
    main() 