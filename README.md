# 🏎️ F1-PitStop-Optimizer: Machine Learning for Race Strategy

This project leverages Machine Learning to transform historical F1 telemetry and race data (2023-2024) into actionable strategic insights. It moves beyond simple data analysis to provide modeling and tactical simulations.

## Key Features
* **EDA (Data Exploration Phase):** During this phase,m the relationships between variables are analysed. This study examines which driver achieved the most Gran Prix victories and accumulates the highest number of points in each of the two years, thereby bein crowned Formula 1 champion. It also looks at tyre lige, the fastest laps recorded during the races, an the timing of pit stops.  
* **Pit Stop Prediction:** A Random Forest clasifier with **91% recall**, identifying optimal pit windows.
* **Explanaible AI (XAI):** Integrated **SHAP** values to decode the decision-making process (e.g., the weight of Track Position vs. Tyre Degradation).
* **Race Simulator:** A stochastic simulation tool that forecasts race scenarios based on position, stint, tyre life, compound, lap time delta, cumulative degradation, position change and race progress.
* **Circuit Clustering:** K-Means clustering to group circuits based on their strategic profile (conservative, standard, aggressive or extreme/special cases).
* **Strategic Heatmaps:** Probabilistic visualizations of pit windows across different tyre compounds (Hard, Medium, Soft).

## Key Insights 
* **Performance:** The model successfully predicts over 91% of tactical pit stops.
* **Strategy:** Data reveals that **Track Position** often outweighs physical tyre wear in modern F1 strategic decision-making.
* **Circuit Patterns:** Distinct clusters separate tracks such as Monaco (high cumulative wear/low degradation rate) and the Chinese Grand Prix (extreme degradation rate) from the Canadian Grand Prix (low cumulative wear/low degradation rate) and the Spanish Grand Prix (balanced degradation).

## Tech Stack
* **Language:** Python 3.14
* **Machine Learning:** Scikit-learn (Random Forest, K-Means, StandardScaler)
* **Explainer:** SHAP (Lundberg & Lee)
* **Visualization:** Seaborn, Matplotlib, AdjustText
* **Data Handling:** Pandas, NumPy

## Project Structure
- `src/`: Main Python scripts/notebooks.
- `images/`: Visualizations (EDA, Heatmaps, Cluster Maps, SHAP plots).
- `requirements.txt`: List of dependencies.

## Conclusion
This project demonstrates how data-driven techniques can be used to predict when individual drivers wil make pit stops. It also identifies circuits with similar characteristics, facilitating the application of comparable race strategies. Finally, it analyses the durability of different tyre types, estimating how many laps they can last for based on their usage conditions.









