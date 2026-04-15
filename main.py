# main.py
from src.config import init_project
from src.preprocessing import load_data, inspect_data
from src.eda import run_exploratory_analysis
from src.machine_learning import train_pit_stop_model, evaluate_and_explain
from src.simulation import predict_driver_pit_stop, run_race_simulation
from src.clustering import run_circuit_clustering
from src.strategy_viz import plot_strategic_heatmap


def main():
    init_project()
    df = load_data('f1_strategy_dataset_v2.csv')
    
    if df is not None:
        # 1. Inspection
        df = inspect_data(df)
        
        # 2. EDA
        run_exploratory_analysis(df)
        
        # 3. Machine Learning Training
        model, scaler_ml, compound_encoder, X_test_scaled, y_test, feature_names = train_pit_stop_model(df)
        
        # 4. ML Evaluation & SHAP
        evaluate_and_explain(model, X_test_scaled, y_test, feature_names)

        # 5. Real-time Pit Stop Prediction Function
        example_telemetry = {
            'Position': 3, 'Stint': 2, 'TyreLife': 30, 'Compound': 1, 
            'LapTime_Delta': 2.5, 'Cumulative_Degradation': 15.0, 
            'Position_Change': -4, 'RaceProgress': 0.85
        }
        predict_driver_pit_stop(model, scaler_ml, feature_names, example_telemetry)

        # 6. Dynamic Race Strategy Simulation
        sim_results = run_race_simulation(model, scaler_ml, feature_names)

        # 7. Circuit Analysis & Strategic Clustering
        circuit_profiles = run_circuit_clustering(df)
        
        # 8. Strategic Probability Heatmap
        plot_strategic_heatmap(model, scaler_ml, feature_names)

        print("\n F1 PitStop Optimizer: Full Pipeline Execution Successful.")

if __name__ == "__main__":
    main()