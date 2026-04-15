# src/config.py
import os
import numpy as np

# Reproducibility
RANDOM_SEED = 42

# Subfolder structure for images
PATHS = {
    'eda': 'images/01_eda',
    'ml': 'images/02_ml_results',
    'clustering': 'images/03_clustering',
    'simulation': 'images/04_simulation'
}

def init_project():
    """Sets the seed and creates the folder structure."""
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
    print("✅ Project environment initialized (Seed set & Folders created).")