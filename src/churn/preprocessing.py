"""Feature engineering shared by the supervised pipelines (Unidad 1 and 2)."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def preprocess_data(df: pd.DataFrame, target_col: str = "churn_30", random_state: int = RANDOM_STATE):
    """One-hot encode, split and scale the dataset for supervised modeling.

    Categorical features are one-hot encoded, the frame is split 80/20 with
    stratification on the target, and numeric features are standardized
    (fit on train only, to avoid leaking test-set statistics).

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    data = pd.get_dummies(df.copy(), columns=CATEGORICAL_FEATURES, drop_first=False)

    exclude_cols = ["customer_id", "churn_30", "churn_60"]
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[NUMERIC_FEATURES]),
        columns=NUMERIC_FEATURES,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[NUMERIC_FEATURES]),
        columns=NUMERIC_FEATURES,
        index=X_test.index,
    )

    dummy_cols = [col for col in feature_cols if col not in NUMERIC_FEATURES]
    X_train_final = pd.concat([X_train_scaled, X_train[dummy_cols]], axis=1)
    X_test_final = pd.concat([X_test_scaled, X_test[dummy_cols]], axis=1)

    print(f"\nPreprocesamiento para target: {target_col}")
    print(f"  - Features totales: {len(feature_cols)}")
    print(f"  - Features numericas: {len(NUMERIC_FEATURES)}")
    print(f"  - Features categoricas (one-hot): {len(dummy_cols)}")
    print(f"  - Train set: {len(X_train):,} ({y_train.mean()*100:.1f}% churn)")
    print(f"  - Test set: {len(X_test):,} ({y_test.mean()*100:.1f}% churn)")

    return X_train_final, X_test_final, y_train, y_test, feature_cols, scaler
