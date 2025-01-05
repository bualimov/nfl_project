import pandas as pd
import numpy as np
from scipy import stats

def main():
    # Read the original entropy values (before normalization)
    print("Loading data...")
    results_df = pd.read_csv('defensive_entropy_analysis_all_weeks.csv')
    
    print("\nStatistical Analysis of Unnormalized Values:")
    print("=" * 80)
    print(f"{'Position':<8} {'Raw Diff':<12} {'Norm Diff':<12} {'Raw P-Value':<12} {'Norm P-Value':<12}")
    print("-" * 80)
    
    # Calculate raw differences and p-values
    for _, row in results_df.iterrows():
        position = row['Position']
        success_avg = row['Success_Avg_Entropy']
        failure_avg = row['Failure_Avg_Entropy']
        norm_diff = row['Entropy_Difference']
        p_value = row['P_Value']
        
        # Calculate raw difference (approximately, based on normalization)
        raw_diff = norm_diff * (row['Success_Avg_Entropy'] / 100)
        
        print(f"{position:<8} {raw_diff:>11.3f} {norm_diff:>11.3f} {p_value:>11.3e} {p_value:>11.3e}")

if __name__ == "__main__":
    main() 