"""Classification evaluation helpers: top-K recall, confusion matrices, threshold sweeps."""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def recall_at_top_k(y_true, y_proba, k_percent: float) -> float:
    """Recall achieved by contacting only the top K% highest-probability customers.

    Answers: "if retention can only reach K% of customers, ranked by risk
    score, what fraction of actual churners does that top K% capture?"
    """
    n_top = int(len(y_true) * k_percent / 100)
    top_indices = np.argsort(y_proba)[-n_top:]
    y_true_top = y_true.iloc[top_indices] if hasattr(y_true, "iloc") else y_true[top_indices]
    return y_true_top.sum() / y_true.sum()


def print_confusion_matrix(y_true, y_pred, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{title}")
    print("-" * 40)
    print("                  Predicho")
    print("                  No Churn  Churn")
    print(f"Real No Churn     {cm[0,0]:>6}    {cm[0,1]:>5}  (TN, FP)")
    print(f"Real Churn        {cm[1,0]:>6}    {cm[1,1]:>5}  (FN, TP)")
    print("-" * 40)
    tn, fp, fn, tp = cm.ravel()
    print(f"Verdaderos Negativos (TN): {tn:,}")
    print(f"Falsos Positivos (FP): {fp:,}")
    print(f"Falsos Negativos (FN): {fn:,}")
    print(f"Verdaderos Positivos (TP): {tp:,}")


def analyze_thresholds(y_true, y_proba, target_name: str, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)) -> None:
    """Print precision/recall/F1/volume for a sweep of decision thresholds.

    Default is 0.5, but the right choice depends on the business ROI of a
    missed churner (false negative) vs. a wasted retention contact (false positive).
    """
    print(f"\nAnalisis de Thresholds - {target_name}")
    print("-" * 70)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Contactados':<15}")
    print("-" * 70)
    for threshold in thresholds:
        y_pred_t = (y_proba >= threshold).astype(int)
        prec = precision_score(y_true, y_pred_t)
        rec = recall_score(y_true, y_pred_t)
        f1 = f1_score(y_true, y_pred_t)
        contacted = y_pred_t.sum()
        marker = " *" if threshold == 0.5 else ""
        print(f"{threshold:<12.1f} {prec:<12.3f} {rec:<12.3f} {f1:<12.3f} {contacted:<15}{marker}")
    print("-" * 70)
    print("* Threshold seleccionado por defecto")
