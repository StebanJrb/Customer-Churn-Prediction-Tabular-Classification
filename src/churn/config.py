"""Constants shared by every unit's pipeline."""

import warnings

import numpy as np
import pandas as pd

RANDOM_STATE = 42

# Columns produced by generate_telecom_dataset(), split by how they need to
# be treated during preprocessing / clustering.
NUMERIC_FEATURES = [
    "tenure", "monthly_charges", "total_charges",
    "num_tickets", "num_complaints", "late_payments",
    "mora_days", "consumption_change",
]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "month"]
CLUSTER_FEATURES = [
    "tenure", "monthly_charges", "num_tickets",
    "num_complaints", "late_payments", "mora_days",
]


def configure_environment():
    """Apply the notebook-style global settings the original scripts set at import time."""
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    np.random.seed(RANDOM_STATE)


def set_plot_style():
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
