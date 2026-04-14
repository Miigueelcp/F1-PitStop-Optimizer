# =================================================================
# F1 PITSTOP OPTIMIZER: Machine Learning for Race Strategy
# Part 1: Environment Setup & Data Preprocessing
# =================================================================

# 1. IMPORT LIBRERIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from matplotlib.ticker import MaxNLocator
from adjustText import adjust_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

# Settings for visualization
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

# We need to create a folder called 'images'
if not os.path.exists('images'):
    os.makedirs('images')
    print("📁 Folder 'images' created successfully.")

# We set a seed to ensure reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 2. LOAD DATASET
def load_data(file_path):
    """Loads the F1 strategy dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully: {file_path}")
        return df
    except FileNotFoundError:
        print("❌ Error: File not found. Please check the file path.")
        return None

df = load_data('f1_strategy_dataset_v2.csv')

# 3. DATA CLEANING & INSPECTION
if df is not None:
    print("\n--- Initial Data Inspection ---")
    print(df.head())
    print(df.tail())

    # Information about the variables
    print('\n Information about the variables')
    print(df.info())

    # Check for missing values and duplicates
    print(f"\nMissing values count: {df.isnull().sum()}")
    print(f"Duplicate rows count: {df.duplicated().sum()}")

    # Target Variable Distribution
    print("\n--- Target Variable (PitNextLap) Distribution ---")
    print(df['PitNextLap'].value_counts(normalize=True) * 100)

    # Statistical Summary
    print("\n--- Statistical Summary ---")
    print(df.describe())


"""
    --- DATA INSIGHTS & CONCLUSIONS ---
    1. Race Duration: Average laps per driver is ~30, suggesting not all sessions 
       reach full race distance (e.g., DNFs or Sprint formats).
    2. Stint Strategy: Average stint count is 2, confirming a standard 
       two-stop strategy as the baseline.
    3. Tyre Life: Compounds show relatively short lifespans, likely due to 
       heavy usage of Soft/Medium compounds over long distances.
    4. Performance Evolution: Lap times generally improve as tyres reach 
       optimal temperature and fuel load decreases.
    5. Overtaking: Position changes are rare but volatile, ranging from 
       -18 to +18 positions depending on incidents.
    6. Class Imbalance: 'PitNextLap' is highly imbalanced; resampling 
       (SMOTE) will be necessary for Machine Learning.
    """


# =================================================================
# Part 2: Exploratory Data Analysis (EDA) & Data Visualization
# =================================================================

def run_exploratory_analysis(df):
    """Generates insights through data visualization."""
    
    # 1. Feature Correlation Study
    plt.figure(figsize=(12, 8))
    corr_data = df.corr(numeric_only=True)
    sns.heatmap(data=corr_data, annot=True, cmap='coolwarm', fmt=".2f",
                vmin=-1, vmax=1, linewidths=1.5, linecolor='white')
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('images/01_correlation_matrix.png')
    plt.show()

    # Observation: Laps and Race Progress are naturally highly correlated. 
    # LapTime shows significant links with tyre degradation and lap deltas.

    # 2. Race Winners Analysis (2023 vs 2024)
    # Identifying the last lap of each race per driver
    last_lap_idx = df.groupby(['Year', 'Race', 'Driver'])['LapNumber'].idxmax()
    df_finals = df.loc[last_lap_idx].copy()
    
    gp_winners = df_finals[df_finals['Position'] == 1]
    win_counts = gp_winners.groupby(['Year', 'Driver']).size().reset_index(name='Wins')
    win_counts = win_counts.sort_values(['Year', 'Wins'], ascending=[True, False])

    plt.figure(figsize=(12, 6))
    sns.barplot(data=win_counts, x='Driver', y='Wins', hue='Year', palette='magma', edgecolor='black')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Grand Prix Wins: 2023 vs 2024', fontsize=15, pad=20)
    plt.xlabel('Pilot', fontsize=12)
    plt.ylabel('Number of Wins', fontsize=12)
    plt.legend(title='Season', loc='upper right')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('images/02_race_wins.png')
    plt.show()
    # Observation: Verstappen dominated 2023 (19 wins), while 2024 shows a more balanced distribution.

    # 3. Championship Points (Total Season Standing)
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    df_finals['Points'] = df_finals['Position'].map(points_map).fillna(0)
    annual_standings = df_finals.groupby(['Year', 'Driver'])['Points'].sum().reset_index()
    annual_standings = annual_standings.sort_values(['Year', 'Points'], ascending=[True, False])

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=annual_standings, x='Year', y='Points', hue='Driver', marker='o')
    plt.title('Total Championship Points per Season', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('World Championship Points', fontsize=12)
    plt.xticks([2023, 2024])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Drivers")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/03_points_standing.png')
    plt.show()
    # Observation: In this picture, we can see that Vertappen won the F1 championship both in 2023 and 2024. The
    # difference in points is vast in 2023.

    # 4. Tyre Life Analysis by Compound
    avg_tyre_life = df.groupby('Compound')['TyreLife'].mean().reset_index()
    order = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    avg_tyre_life['Compound'] = pd.Categorical(avg_tyre_life['Compound'], categories=order, ordered=True)
    avg_tyre_life = avg_tyre_life.sort_values('Compound')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=avg_tyre_life, x='Compound', y='TyreLife', palette='magma')
    plt.title('Average Tyre Life by Compound', fontsize=15)
    plt.xlabel('Type of Tyre', fontsize = 12)
    plt.ylabel('Average Laps per Stint', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/04_tyre_life.png')
    plt.show()
    # Observation : Tyres with a longer average lifespan are hard tyres, followed by intermediante tyres (used in
    # raininy conditions) that has more average life is hard tyres, whose follor intermediate (used in raining days), medium tyres,
    # soft tyres and and wet tyres.

    # 5. Fastest Lap Times by GP
    fastest_laps_idx = df.groupby(['Race', 'Year'])['LapTime (s)'].idxmin()
    fastest_laps = df.loc[fastest_laps_idx].copy()

    lap_counts = fastest_laps.sort_values(['Race', 'Year'])

    plt.figure(figsize=(12, 10))
    ax = sns.scatterplot(data= lap_counts, x='LapTime (s)', y='Race', hue= 'Year', style= 'Year', s = 150,
                     palette= 'magma', edgecolor = 'black', zorder = 3)
    
    texts = []
    for i in range(len(lap_counts)):
        row = lap_counts.iloc[i]
        texts.append(plt.text(row['LapTime (s)'], row['Race'], row['Driver'], fontsize=8, fontweight='bold'))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    min_t = lap_counts['LapTime (s)'].min()
    max_t = lap_counts['LapTime (s)'].max()
    plt.xlim(min_t - 5, max_t + 10)

    plt.title('Fastest Lap Time per GP and Year', fontsize=16, pad=20)
    plt.xlabel('Lap Time (s)', fontsize=12)
    plt.ylabel('GP', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig('images/05_fastest_laps.png')
    plt.show()
    # Observation : In this picture we can see the fastests lap time of the different pilots per Gran Prix and year. There are
    # a few differences between them, with the fastest lap time being about 69 seconds in the Austrian Gran Prix and
    # the slowest lap time being about 109 seconds in the Belgian Grand Prix.
    
    # 6. Pit Stop Timing Analysis
    df_pits = df[df['PitStop'] == 1]
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df_pits, x='RaceProgress', hue='Compound', fill=True, common_norm=False, palette='viridis', alpha=0.3)
    plt.title('Race Progress vs. Pit Stop Density', fontsize=16)
    plt.xlabel('Race Progress (%)', fontsize=12)
    plt.xlabel('Pit Stop Density', fontsize=12)
    plt.axvline(0.33, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(0.66, color='gray', linestyle='--', alpha=0.3)
    plt.xlim(0, 1)
    plt.xticks([0, 0.25, 0.5, 0.75, 1], ['Start', '25%', 'Halfway', '75%', 'Finish'])
    plt.grid(True, axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig('images/06_pit_stop_timing.png')
    plt.show()
    # Observation: This picture shows the stage at which pit stops are made to fit the different tyres during the race. 
    # As we can see, the soft tyres are fitted at around the 65 % mark, as these are used for the final sprint to help
    # the cars accelerate. 

# Run the EDA
run_exploratory_analysis(df)



# =================================================================
# Part 3: Machine Learning - Pit Stop Prediction Model
# 3.1: Preprocessing, Data Balancing (SMOTE) & Training
# =================================================================

def train_pit_stop_model(df):
    """
    Handles data encoding, feature scaling, class imbalance, 
    and trains the Random Forest classifier.
    """

    # 1. Feature Encoding
    # We convert categorical 'Compound' names into numerical labels for the model
    df_ml = df.copy()
    label_encoder = LabelEncoder()
    df_ml['Compound'] = label_encoder.fit_transform(df_ml['Compound'])

    # 2. Feature Selection
    # These variables represent the tactital state of the car and race
    features_names = [
        'Position', 'Stint', 'TyreLife', 'Compound', 'LapTime_Delta', 
        'Cumulative_Degradation', 'Position_Change', 'RaceProgress'
    ]
    X = df_ml[features_names]
    y = df_ml['PitNextLap']

    # 3. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state= RANDOM_SEED
    )

    # 4. Feature Scaling
    # Standarization is crucial as features have different units
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Handling Class Imbalance (SMOTE)
    # Pit stops are rare events. SMOTE oversamples the minority class to prevent the model from being
    # biased towards 'No Pit Stop'
    smote = SMOTE(random_state= RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

    # 6. Model Training: Random Forest
    # Using 150 estimators and limited depth to balance precision and generalization
    model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state= RANDOM_SEED)
    model.fit(X_resampled, y_resampled)

    return model, scaler, label_encoder, X_test_scaled, y_test, features_names

# Execute the training pipeline
model, scaler_ml, compound_encoder, X_test_scaled, y_test, feature_names = train_pit_stop_model(df)



# =================================================================
# Part 3.2: Model Evaluation & Performance Metrics
# =================================================================

def evaluate_and_explain(model, X_test_scaled, y_test, feature_names):
    """
    Comprehensive evaluation: Calculates performance metrics and 
    generates SHAP values to explain model decisions.
    """

    # 1. Performance Metrics (Accuracy, Recall, F1)
    y_pred = model.predict(X_test_scaled)
    print("\n📊 MODEL PERFORMANCE REPORT:")
    print(classification_report(y_test, y_pred))

    # High Recall (0.91) for Class 1 (Pit Stop) is critical. It means the model captures 91% of real-world tactical stops, 
    # which is the priority for a race strategist

    # 2. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Pit', 'Pit Stop'])

    disp.plot(cmap='Blues', values_format='d', ax=ax)
    ax.set_title('Confusion Matrix: Pit Stop Prediction', pad=20)
    ax.grid(False)
    plt.savefig('images/07_confusion_matrix.png')
    plt.show()
    plt.close()

    # 3. SHAP (Explanaible AI)
    # We convert back to DataFrame to provide feature names to SHAP
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    X_sample = X_test_df.sample(300, random_state= RANDOM_SEED)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample, check_additivity=False)

    # SHAP Beeswarm Plot
    plt.figure(figsize=(12, 8))
    # Index [:,:,1] refers to the probability of the "Pit Stop" class
    shap.plots.beeswarm(shap_values[:,:,1], max_display=10, show=False)

    plt.title('Strategic Decision Analysis: Why does the model predict a Pit Stop?', fontsize=14, pad=25)
    plt.xlabel('Impact on Probability (SHAP Value)')
    plt.tight_layout()
    plt.savefig('images/08_shap_beeswarm.png')
    plt.show()
    plt.close()

    """
    --- STRATEGIC INSIGHTS FROM SHAP ---
    The model acts as a rational strategist. It ignores the noise of absolute poisitioning
    and focuses on the degradation-to-performance ratio, while bearing in mind the reamining
    race distance in order to avoid unnceserarry stops
    """
    
# Running evaluation and SHAP analysis
evaluate_and_explain(model, X_test_scaled, y_test, feature_names=feature_names)



# =================================================================
# Part 4: Strategic Decision Support Tools
# 4.1: Real-time Pit Stop Prediction Function
# =================================================================

def predict_driver_pit_stop(position, stint, tyre_life, compound, lap_time_delta, 
                            cumulative_deg, position_change, race_progress):
    
    """
    Simulates a pit wall decision by calculating the probability 
    of a pit stop for a specific driver state.
    """

    # 1. Prepare data in the exact order the model expects. We use the feature list from the training phase
    driver_data = pd.DataFrame([[
        position, stint, tyre_life, compound, lap_time_delta, 
        cumulative_deg, position_change, race_progress
    ]], columns=feature_names)

    # 2. Scale the telemetry data using the pre-fitted scaler
    driver_scaled = scaler_ml.transform(driver_data)

    # 3. Make the predictions
    prediction = model.predict(driver_scaled)
    probabilities = model.predict_proba(driver_scaled)
    pit_prob = probabilities[0][1]

    print(f"\n--- 🏁 REAL-TIME TELEMETRY ANALYSIS ---")
    if prediction[0] == 1:
        print("🔴 STATUS: BOX, BOX! Enter the pits on the next lap.")
    else:
        print("🟢 STATUS: STAY OUT. Maintain track position.")
    
    print(f"📈 Pit Stop Probability: {pit_prob:.2%}")
    
    # Tactical Note for the Engineer
    if 0.40 <= pit_prob <= 0.60:
        print("⚠️ TACTICAL ALERT: Critical decision window. Tyre life is at the edge.")
    elif pit_prob > 0.85:
        print("🚨 URGENT: High degradation detected. Pit stop mandatory.")


# --- EXAMPLE CASE ---
# Scenario : A driver in P3, struggling on an old stint at 85 % race progress.
print("\nRunning example scenario...")
predict_driver_pit_stop(
    position=3, stint=2, tyre_life=30, compound=1, 
    lap_time_delta=2.5, cumulative_deg=15.0, 
    position_change=-4, race_progress=0.85
)


# =================================================================
# Part 4.2: Dynamic Race Strategy Simulation
# =================================================================

def get_pit_probability(state_dict, scaler, model, feature_cols):

    """
    Helper function to get a single probability value from a state dictionary.
    """

    # Convert dict to DataFrame with correct column order
    df_input = pd.DataFrame([state_dict])[feature_cols]

    # Scale and predict
    df_scaled = scaler.transform(df_input)
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    return pred, prob


def run_race_simulation(total_laps=65, pit_threshold=0.7):

    """
    Simulates a full race lap-by-lap, predicting pit stops 
    and resetting car state after a 'BOX' event.
    """

    history = []

    # Initial State (Race Start)
    state = {
        'Position': 5,
        'Stint': 1,
        'TyreLife': 1,
        'Compound': 1, 
        'LapTime_Delta': 0.0,
        'Cumulative_Degradation': 0.0,
        'Position_Change': 0,
        'RaceProgress': 0.0
    }

    print(f"\n--- Starting Race Simulation ({total_laps} Laps) ---")

    for lap in range(1, total_laps + 1):
        # Update progress
        state['RaceProgress'] = lap / total_laps

        # Get probability from model
        pred, prob = get_pit_probability(state, scaler_ml, model, feature_names)

        # Log current lap data
        history.append({
            'Lap': lap,
            'Probability': prob,
            'Stint': state['Stint'],
            'TyreLife': state['TyreLife'],
            'PitPerformed': 0
        })

        # Decision Logic: If prob > threshold, perform PIT STOP (unless it's the very end)
        if prob >= pit_threshold and lap < total_laps - 2:
            history[-1]['PitPerformed'] = 1
            
            # --- PIT STOP RESET ---
            state['Stint'] += 1
            state['TyreLife'] = 1
            
            # Compound swap logic (stochastic selection of a different compound)
            current_comp = state['Compound']
            next_comp = np.random.choice([c for c in [0, 1, 2] if c != current_comp])
            state['Compound'] = next_comp
            
            # Reset wear metrics
            state['Cumulative_Degradation'] = 0.0
            state['LapTime_Delta'] = 0.0
            
            print(f"Lap {lap}: 🔴 BOX BOX BOX! (Prob: {prob:.2%}) -> Switching to Compound {next_comp}")

        else:
            # --- ON-TRACK EVOLUTION ---
            state['TyreLife'] += 1
            # Stochastic degradation (adds realism)
            state['Cumulative_Degradation'] += np.random.uniform(0.1, 0.4)
            state['LapTime_Delta'] += np.random.uniform(0.02, 0.15)
            state['Position_Change'] = np.random.randint(-1, 2)
    
    return pd.DataFrame(history)


# Execute Simulation
sim_results = run_race_simulation(total_laps=65, pit_threshold=0.7)

# --- VISUALIZATION ---
plt.figure(figsize=(14, 6))

# Probability Line
plt.plot(sim_results['Lap'], sim_results['Probability'], 
         color='#1f77b4', lw=2.5, label='Pit Stop Probability', zorder=2)

# Shading Stints
for stint_num in sim_results['Stint'].unique():
    subset = sim_results[sim_results['Stint'] == stint_num]
    plt.fill_between(subset['Lap'], 0, subset['Probability'], alpha=0.1)
    # Add text label for each stint
    plt.text(subset['Lap'].mean(), 0.05, f"STINT {stint_num}", 
             ha='center', fontweight='bold', alpha=0.6)

# Marking Pit Stops
pit_events = sim_results[sim_results['PitPerformed'] == 1]
plt.scatter(pit_events['Lap'], pit_events['Probability'], 
            color='red', s=120, edgecolor='black', label='Pit Stop Executed', zorder=3)

# Critical Threshold Line
plt.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Critical Threshold (0.7)')

plt.title("Race Strategy Simulation: Probability Evolution vs. Tactical Window", fontsize=16, pad=20)
plt.xlabel("Race Lap", fontsize=12)
plt.ylabel("Prediction Probability", fontsize=12)
plt.ylim(0, 1.05)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('images/09_race_simulation.png')
plt.show()




# =================================================================
# Part 5: Circuit Analysis & Strategic Clustering
# =================================================================


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
    plt.savefig('images/10_elbow_method.png')
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
    plt.savefig('images/11_circuit_strategic_map.png')
    plt.show()

    return circuit_stats

# Run the Clustering Analysis
circuit_profiles = run_circuit_clustering(df)



# =================================================================
# Part 6: Strategic Probability Heatmap
# =================================================================

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
    plt.savefig('images/12_strategic_heatmap.png')
    plt.show()

# Final Execution
plot_strategic_heatmap(model, scaler_ml, feature_names)