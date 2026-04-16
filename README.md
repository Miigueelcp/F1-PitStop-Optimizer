# 🏎️ F1-PitStop-Optimizer: Machine Learning for Race Strategy

This project leverages **Machine Learning** to transform historical F1 telemetry and race data (2023-2024) into actionable strategic insights. It moves beyond simple data analysis to provide modeling and tactical simulations.

## Key Features
* **Exploratory Data Analysis (EDA):** A deep dive into race variables, including driver performance (wins/points), tyre life cycles, fastest laps and historical pit stop timing.
* **Pit Stop Prediction:** A Random Forest clasifier with **91% recall**, specifically tuned to identify optimal pit windows before the performance 'cliff'.
* **Explanaible AI (XAI):** Integrated **SHAP** values to decode the model's 'black box', revealing the weight of Track Position vs. Tyre Degradation.
* **Race Strategy Simulator:** A stochastic simulation tool that forecasts race scenarios based on position, stint, tyre life, compound, lap time delta, cumulative degradation, position change and race progress.
* **Circuit Clustering:** K-Means clustering (optimised via the elbow method) groups global circuits into four strategic profiles: Conservative, Standard, Aggresive and Extreme.
* **Strategic Heatmaps:** Probabilistic visualizations of pit windows across different tyre compounds (Hard, Medium, Soft).

## Dataset & Variables
The model is trained on over 52,000 samples of professional racing telemetry. For a detailed breakdown of the 16 variables and engineered features, see the [Data Dictionary](docs/data_dictionary.md).

## Key Insights 
* **Performance:** The model successfully predicts over 91% of tactical pit stops.
* **Strategy:** The model has learned that a driver should stay out even with high tire wear if the lap times have not 'fallen of the cliff' yet.
* **Circuit Patterns:** Distinct clusters separate 'Tyre Killers' like the Chinese GP (extreme degradation) from tactical anomalies like Monaco (high wear but low degradation rate) and balanced tracks like Barcelona.

## Data Preprocessing & Engineering
* **Feature Scaling:** Numerical variables were standardized using StandardScales to normalize their distributions (mean = 0, standard deviation = 1). This prevents variables with larger magnitudes from dominating the model.
*  **Class Imbalance Handling (SMOTE):** The target variable 'PitNextLap' is naturally imbalanced, as pit stops are relatively rare events. To addres this, SMOTE (Synthetic Minority Over-sampling Technique) was applied to generate synthetic samples of the minority class and improve model recall.
*  **Feature Engineering:** Custom features were engineered to represent the underlying dynamics of each circuit. These variables capture tyre degradation rates, lap time variability, and performance evolution, enabling the application of clustering algorithms (K-Means) to identify circuits with similiar strategic profiles.

## Tech Stack
* **Language:** Python 3.14
* **Machine Learning:** Scikit-learn (Random Forest, K-Means, StandardScaler)
* **Imbalance Handling:** imbalanced-learn (SMOTE)
* **Explainer:** SHAP (Lundberg & Lee)
* **Visualization:** Seaborn, Matplotlib, AdjustText
* **Data Handling:** Pandas, NumPy


## Project Structure
- `main.py`: The entry point of the application. Ir orchestrates the entire pipeline from data loading to race simulation.
- **`src/`**: Specialized modules containing the core logic:
    * `config.py`: Global settings, reproducibility seeds, and automated directory management.
    * `preprocessing.py`: Functions for data loading, cleaning and initial statistical inspection.
    * `eda.py`: A comprehensive visualization suite for analyzing race trends and driver performance.
    * `machine_learning.py`: Model training (Random Forest), class balancing (SMOTE), and Explainable AI (SHAP).
    * `simulation.py`: Dynamic race engine and lap-by-lap probability generator.
    * `clustering.py`: K-Means implementation to identify and profile circuits."
    * `strategy_viz.py`: Advanced visual tool, including the Strategic Probability Heatmaps.
* **`images/`**: Resulting assets organized by analysis phase:
    * `01_eda/`: Correlation matrices, season wins, points standing, tire life cicles, fastest laps and pit stop timing.
    * `02_ml_results/`: Confusion matrices and feature impact plots (SHAP).
    * `03_clustering/`: Elbow method analysis and the F1 Circuit Strategic Map.
    * `04_simulation/`: Race simulation logs and a strategic tyre heat map.
- `docs/`: Technical documentation, including the detailed [Data Dictionary](docs/data_dictionary.md).
- `data/`: Local storage for the telemetry dataset (e.g., `f1_strategy_dataset_v2.csv`).
- `requirements.txt`: Environment dependencies.


## Conclusion
This project shows how data-driven techniques can predict when individual drivers will make pit stops. It also identifies circuits with similiar characteristics, enabling comparable race strategies to be applied. Finally, it analyses the durability of different tyre types and estimates how many laps that can last based on their usage conditions.









