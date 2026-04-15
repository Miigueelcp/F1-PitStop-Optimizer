# src/preprocessing.py
import pandas as pd

def load_data(file_path):
    """Loads the F1 strategy dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully: {file_path}")
        return df
    except FileNotFoundError:
        print("❌ Error: File not found. Please check the file path.")
        return None

def inspect_data(df):
    """Performs initial data inspection and prints stats."""

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

    return df

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
