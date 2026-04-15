# src/strategy_viz.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PATHS

def plot_strategic_heatmap(model, scaler, feature_cols):

    """
    Creates a probability grid to visualize the optimal pit stop windows
    for Hard, Medium, and Soft compounds over a 60-lap span.
    """

    print("\n--- Generating Strategic Heatmap ---")
    
    # 1. Setup Simulation Grid
    max_laps = 60
    laps = np.arange(1, max_laps + 1)
    compound_ids = [0, 1, 2] # 0: HARD, 1: MEDIUM, 2: SOFT
    compound_names = ['HARD', 'MEDIUM', 'SOFT']
    
    # Create a virtual grid (Typical mid-field scenario)
    grid_data = []
    for comp in compound_ids:
        for v in laps:
            grid_data.append({
                'Position': 10, 
                'Stint': 1, 
                'TyreLife': v, 
                'Compound': comp,
                # Simulating quadratic lap time loss and linear degradation
                'LapTime_Delta': 0.02*v + 0.002*v**2, 
                'Cumulative_Degradation': 0.1 * v,
                'Position_Change': 0, 
                'RaceProgress': v / max_laps
            })
    
    # Convert to DataFrame with correct column order
    df_virtual = pd.DataFrame(grid_data)[feature_cols]
    
    # 2. Batch Prediction
    # We use the previously fitted scaler to maintain consistency
    X_virtual_scaled = scaler.transform(df_virtual)
    probabilities = model.predict_proba(X_virtual_scaled)[:, 1]
    
    # Reshape probabilities into a matrix (Compounds x Laps)
    prob_matrix = probabilities.reshape(len(compound_ids), len(laps))
    
    # 3. Visualization
    plt.figure(figsize=(16, 7))
    
    # RdYlGn_r: Red (Stop), Yellow (Alert), Green (Stay Out)
    ax = sns.heatmap(
        prob_matrix, 
        cmap='RdYlGn_r', 
        annot=False,
        xticklabels=5, 
        yticklabels=compound_names,
        cbar_kws={'label': 'Pit Stop Probability', 'pad': 0.02},
        linewidths=0.1,
        linecolor='#f0f0f0'
    )
    
    # 4. Critical Threshold Marker (70%)
    # Draw a white marker where the probability first crosses the 0.7 threshold
    for i in range(len(compound_ids)):
        critical_idx = np.where(prob_matrix[i] >= 0.7)[0]
        if len(critical_idx) > 0:
            first_v = critical_idx[0]
            plt.plot(first_v + 0.5, i + 0.5, marker='o', color='white', 
                     markeredgecolor='black', markersize=10, label='Critical Window' if i==0 else "")
    
    plt.title('Strategic Windows: Pit Stop Probability by Compound', fontsize=18, fontweight='bold', pad=25)
    plt.xlabel('Tyre Life (Laps)', fontsize=13)
    plt.ylabel('Tyre Compound', fontsize=13)
    
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{PATHS['simulation']}/12_strategic_heatmap.png")
    plt.show()

