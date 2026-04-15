# src/clustering.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text
from src.config import RANDOM_SEED, PATHS

def run_circuit_clustering(df):

    """
    Groups circuits into strategic profiles based on degradation 
    rates, consistency, and tactical pit windows.
    """

    print("\n--- Starting Circuit Clustering (K-Means) ---")

    # 1. Aggregate Statistics per Race
    pit_stops_data = df[df['PitStop'] == 1]
    race_progress_stats = pit_stops_data.groupby('Race')['RaceProgress'].mean()

    # Define circuit-level metrics
    circuit_stats = df.groupby('Race').agg({
        'LapTime_Delta': ['mean', 'std'],
        'TyreLife': 'mean',
        'Cumulative_Degradation': 'max'
    }).reset_index()

    # Flatten multi-index columns
    circuit_stats.columns = ['Race', 'LapTime_mean', 'LapTime_std', 'TyreLife', 'Cumulative_Degradation']

    # 2. Feature Engineering for Clustering
    circuit_stats['RaceProgress'] = circuit_stats['Race'].map(race_progress_stats)
    circuit_stats['Deg_Rate'] = circuit_stats['LapTime_mean'] / circuit_stats['TyreLife']
    circuit_stats['Consistency'] = circuit_stats['LapTime_std']
    circuit_stats['Deg_Total_per_Lap'] = circuit_stats['Cumulative_Degradation'] / circuit_stats['TyreLife']

    # Clean missing values to maintain data integrity
    circuit_stats = circuit_stats.dropna(subset=['Deg_Rate', 'Deg_Total_per_Lap', 'RaceProgress', 'Consistency'])

    # 3. Scaling for K-Means
    features_clustering = ['Deg_Rate', 'Deg_Total_per_Lap', 'RaceProgress', 'Consistency']
    # Use a specific scaler for clustering to avoid mixing with the ML model scaler
    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(circuit_stats[features_clustering])

    # 4. Elbow Method: Finding the Optimal K
    inertia = []
    k_range = range(2, 8)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state= RANDOM_SEED, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertia, marker='o', linestyle='--', color='black')
    plt.title('Elbow Method: Finding Optimal Number of Clusters', fontsize=14)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    plt.savefig(f"{PATHS['clustering']}/10_elbow_method.png")
    plt.show()

    # 5. Final Clustering (k=4)
    print("Assigning circuits to 4 strategic clusters")
    kmeans_final = KMeans(n_clusters=4, random_state= RANDOM_SEED, n_init=10)
    circuit_stats['Cluster'] = kmeans_final.fit_predict(X_scaled)

    # Label clusters logically based on Aggression (Deg_Rate)
    cluster_order = circuit_stats.groupby('Cluster')['Deg_Rate'].mean().sort_values().index
    naming_logic = {
        cluster_order[0]: "Conservative Strategy",
        cluster_order[1]: "Standard Strategy",
        cluster_order[2]: "Aggressive Strategy",
        cluster_order[3]: "Extreme / Special Cases"
    }
    circuit_stats['Profile'] = circuit_stats['Cluster'].map(naming_logic)

    # 6. STRATEGIC MAP VISUALIZATION
    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")

    scatter = sns.scatterplot(
        data=circuit_stats,
        x='Deg_Rate',
        y='Deg_Total_per_Lap',
        hue='Profile',
        size='Consistency',
        sizes=(150, 600),
        palette='viridis',
        edgecolor='black',
        alpha=0.75,
        zorder=3
    )

    # Dynamic Labels with adjust_text
    texts = []
    for i, row in circuit_stats.iterrows():
        label = row['Race'].replace('Grand Prix', 'GP')
        texts.append(plt.text(row['Deg_Rate'], row['Deg_Total_per_Lap'], label, 
                              fontsize=9, fontweight='bold', color='#333333', zorder = 4))

    adjust_text(texts,
                arrowprops=dict(arrowstyle='->', color='red', lw=0.6, alpha=0.6),
                expand_points=(1.5, 1.5),
                force_text=0.5,
                add_objects=[scatter])

    plt.title('F1 Circuit Strategic Map', fontsize=18, pad=25, fontweight='bold')
    plt.xlabel('Degradation Rate (Asphalt Aggressiveness)', fontsize=12)
    plt.ylabel('Total Wear per Lap', fontsize=12)
    plt.legend(title="Circuit Profile", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    plt.grid(True, linestyle='--', alpha=0.3, zorder=0)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(f"{PATHS['clustering']}/11_circuit_strategic_map.png")
    plt.show()

    return circuit_stats

