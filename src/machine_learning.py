# src/machine_learning.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.config import RANDOM_SEED, PATHS


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

    plt.savefig(f"{PATHS['ml']}/07_confusion_matrix.png")
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
    plt.savefig(f"{PATHS['ml']}/08_shap_beeswarm.png")
    plt.show()
    plt.close()

    """
    --- STRATEGIC INSIGHTS FROM SHAP ---
    The model acts as a rational strategist. It ignores the noise of absolute positioning
    and focuses on the degradation-to-performance ratio. Furthermore, it accounts for the remaining race
    distance to prevent inefficiente or unnecesary late-race-stops, ensuring every pit call maximizes
    the tacticar advantage.
    """