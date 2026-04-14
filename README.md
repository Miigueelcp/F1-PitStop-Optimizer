# 🏎️ F1-PitStop-Optimizer: Machine Learning for Race Strategy

This project leverages Machine Learning to transform historical F1 telemetry and race data (2023-2024) into actionable strategic insights. It moves beyond simple data analysis to provide modeling and tactical simulations.

## Key Features
* ** Exploratory Data Analysis (EDA):** A deep five into race variables, including driver performance (wins/points), tyre life cycles, fastest laps and historical pit stop timing.
* **Pit Stop Prediction:** A Random Forest clasifier with **91% recall**, specifically tuned to identify optimal pit windows before the performance 'cliff'.
* **Explanaible AI (XAI):** Integrated **SHAP** values to decode the model's 'black box', revealing the weight of Track Position vs. Tyre Degradation.
* **Race Strategy Simulator:** A stochastic simulation tool that forecasts race scenarios based on position, stint, tyre life, compound, lap time delta, cumulative degradation, position change and race progress.
* **Circuit Clustering:** K-Means clustering (optimised via the elbow method) groups global circuits into four strategic profiles: Conservative, Standard, Aggresive and Extreme.
* **Strategic Heatmaps:** Probabilistic visualizations of pit windows across different tyre compounds (Hard, Medium, Soft).

## Key Insights 
* **Performance:** The model successfully predicts over 91% of tactical pit stops.
* **Strategy:**The model has learned that a driver should stay out even with high tire wear if the lap times have not 'fallen of the cliff' yet.
* **Circuit Patterns:** Distinct clusters separate 'Tyre Killers' like the Chinese GP (extreme degradation) from tactical anomalies like Monaco (high wear but low degradation rate) and balanced tracks like Barcelona.

## Tech Stack
* **Language:** Python 3.14
* **Machine Learning:** Scikit-learn (Random Forest, K-Means, StandardScaler)
* **Explainer:** SHAP (Lundberg & Lee)
* **Visualization:** Seaborn, Matplotlib, AdjustText
* **Data Handling:** Pandas, NumPy

## Project Structure
- `src/`: Main Python scripts containing the ML pipeline and simulation logic.
- `images/`: Visualizations (EDA, Heatmaps, Cluster Maps, SHAP plots).
- `images/`: Source telemetry and cleaned dataset.
- `requirements.txt`: Environment dependencies.

## Conclusion
This project demonstrates how data-driven techniques can be used to predict when individual drivers will make pit stops. It also identifies circuits with similar characteristics, facilitating the application of comparable race strategies. Finally, it analyses the durability of different tyre types, estimating how many laps they can last for based on their usage conditions.









