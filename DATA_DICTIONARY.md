# 📖 Data Dictionary - F1 Strategy Optimizer

This document provides a detailed description of the 16 variables used in the pit stop prediction model. The original data is taken from the **F1 Strategy Dataset** (Aadi Gupta, Kaggle), with additional feature engineering.

## 1. Dataset Variables

| Variable | Type | Description |
| :--- | :--- | :--- |
| **Year** | Integer | The Formula 1 world championship season. |
| **Race** | Categorical | The specific Grand Prix event name. |
| **Driver** | Categorical | Unique three-letter driver code (e.g., VER, HAM, LEC). |
| **LapNumber** | Integer | The current lap index within the race. |
| **Position** | Integer | The driver's track order during the specific lap. |
| **LapTime (s)** | Float | Total time in seconds to complete the lap. |
| **TyreLife** | Integer | Total number of laps completed on the current tire set. |
| **Normalized_TyreLife** | Float | Tire life normalized whitin the stint duration (scaled 0 to 1). |
| **Compound** | Categorical | Type of tyre used. |
| **LapTime_Delta** | Float | Change in lap time performance compared to the previous lap. |
| **Cumulative_Degradation** | Float | Total accumulated performance drop due to tyre wear. |
| **Position_Change** | Integer | Net gain or loss of possition compared to the previous lap. |
| **RaceProgress** | Float | Fraction of total race distance completed (0.0 → 1.0). |
| **PitStop** | Binay | Boolean (1/0) showing whether or not the driver entered the pits on that lap. |
| **PitNextLap** | Binary | **Target Variable:** Predicts if the driver will pit on the following lap.|

---

## 2. Feature Engineering

For the **Circuit Strategy Map**, data was aggregated by circuit ('Race') using the following engineered metrics to define the unique characteristics of each track.

| Variable | Formula | Type | Description |
| :--- | :--- | :--- | :--- |
| **Deg_Rate** | `LapTime_mean / TyreLife_mean` | Float | **Degradation Rate:** Measues the loss of performance (in seconds) per lap as the tyre ages. It quantifies the aggresiveness of the asphalt. |
| **Consistency** | `LapTime_std` | Float | **Consistency:** The standard deviation of lap times. High values suggest unpredictable conditions, traffic impact or sensitive tyre conditions. |
| **Deg_Total_per_Lap** | `Cumulative_Degradation_max / TyreLife_mean` | Float | **Total War Index:** This evaluates the maximum physical impact on the compoung in relation to the stint length. |




