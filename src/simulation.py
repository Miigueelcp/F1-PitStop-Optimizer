# src/simulation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import PATHS, RANDOM_SEED

def predict_driver_pit_stop(model, scaler, feature_names, telemetry_data):
    
    """
    Simulates a pit wall decision by calculating the probability 
    of a pit stop for a specific driver state.
    """

    # 1. Prepare data in the exact order the model expects. We use the feature list from the training phase
    driver_data = pd.DataFrame([telemetry_data])[feature_names]

    # 2. Scale the telemetry data using the pre-fitted scaler
    driver_scaled = scaler.transform(driver_data)

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



def get_pit_probability(state_dict, scaler, model, feature_names):

    """
    Helper function to get a single probability value from a state dictionary.
    """

    # Convert dict to DataFrame with correct column order
    df_input = pd.DataFrame([state_dict])[feature_names]

    # Scale and predict
    df_scaled = scaler.transform(df_input)
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    return pred, prob


def run_race_simulation(model, scaler, feature_names, total_laps=65, pit_threshold=0.7):

    """
    Simulates a full race lap-by-lap, predicting pit stops 
    and resetting car state after a 'BOX' event.
    """

    print(f"\n--- Starting Race Simulation ({total_laps} Laps) ---")

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

    for lap in range(1, total_laps + 1):
        # Update progress
        state['RaceProgress'] = lap / total_laps

        # Get probability from model
        pred, prob = get_pit_probability(state, scaler, model, feature_names)

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
    
    sim_df = pd.DataFrame(history)
    _plot_simulation(sim_df, pit_threshold)
    
    return sim_df

def _plot_simulation(sim_results, threshold):
    """Generates the simulation chart."""

    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 6))

    # Probability Line
    plt.plot(sim_results['Lap'], sim_results['Probability'], color='#1f77b4', lw=2.5, label='Pit Stop Probability', zorder=2)

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
    plt.savefig(f"{PATHS['simulation']}/09_race_simulation.png")
    plt.show()