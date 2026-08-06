"""Unidad 1 baseline: Logistic Regression vs. Random Forest for churn classification."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from churn.config import RANDOM_STATE


def train_and_evaluate_models(X_train, X_test, y_train, y_test, target_name: str, random_state: int = RANDOM_STATE) -> dict:
    """Train Logistic Regression (baseline) and Random Forest (ensemble), report metrics.

    Both use class_weight='balanced' to compensate for churn being the
    minority class. Returns a dict keyed by model name plus 'feature_importance'.
    """
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    print(f"\n{'='*40}")
    print(f"Modelo: Regresion Logistica - {target_name}")
    print(f"{'='*40}")

    lr_model = LogisticRegression(
        class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=random_state
    )
    cv_scores_lr = cross_val_score(lr_model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores_lr.mean():.3f} (+/- {cv_scores_lr.std():.3f})")

    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_proba_lr = lr_model.predict_proba(X_test)[:, 1]

    results["logistic_regression"] = {
        "model": lr_model,
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr),
        "recall": recall_score(y_test, y_pred_lr),
        "f1": f1_score(y_test, y_pred_lr),
        "roc_auc": roc_auc_score(y_test, y_proba_lr),
        "pr_auc": average_precision_score(y_test, y_proba_lr),
        "y_pred": y_pred_lr,
        "y_proba": y_proba_lr,
    }
    _print_metrics(results["logistic_regression"])

    print(f"\n{'='*40}")
    print(f"Modelo: Random Forest - {target_name}")
    print(f"{'='*40}")

    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=5,
        class_weight="balanced", random_state=random_state, n_jobs=-1,
    )
    cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std():.3f})")

    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    results["random_forest"] = {
        "model": rf_model,
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
        "f1": f1_score(y_test, y_pred_rf),
        "roc_auc": roc_auc_score(y_test, y_proba_rf),
        "pr_auc": average_precision_score(y_test, y_proba_rf),
        "y_pred": y_pred_rf,
        "y_proba": y_proba_rf,
    }
    _print_metrics(results["random_forest"])

    print("\nTop 10 Variables mas Importantes (Random Forest):")
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    results["feature_importance"] = feature_importance

    return results


def _print_metrics(model_results: dict) -> None:
    print("\nMetricas en Test Set:")
    print(f"  Accuracy:  {model_results['accuracy']:.3f}")
    print(f"  Precision: {model_results['precision']:.3f}")
    print(f"  Recall:    {model_results['recall']:.3f}")
    print(f"  F1-Score:  {model_results['f1']:.3f}")
    print(f"  ROC-AUC:   {model_results['roc_auc']:.3f}")
    print(f"  PR-AUC:    {model_results['pr_auc']:.3f}")
