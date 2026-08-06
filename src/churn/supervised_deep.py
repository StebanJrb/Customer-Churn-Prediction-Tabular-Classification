"""Unidad 2 - Profundizacion en aprendizaje supervisado.

Extiende el baseline de Unidad 1 (Regresion Logistica + Random Forest) con
un tercer algoritmo (Arbol de Decision), analisis de generalizacion
(overfitting/underfitting) y optimizacion de hiperparametros via Grid Search.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from churn.config import RANDOM_STATE


def train_and_evaluate_models(X_train, X_test, y_train, y_test, random_state: int = RANDOM_STATE) -> dict:
    """Entrena y evalua Regresion Logistica, Arbol de Decision y Random Forest.

    Incluye validacion cruzada (k=5) y prediccion sobre train y test (esta
    ultima se usa despues en analyze_generalization para detectar overfitting).

    Returns
    -------
    dict
        Clave = nombre del modelo, valor = dict con 'model', 'y_pred_train',
        'y_pred', 'y_proba', 'cv_scores'.
    """
    print("ENTRENAMIENTO DE MODELOS DE CLASIFICACION")
    print("=" * 80)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results = {}

    # ------------------------------------------------------------------
    # 1. Regresion Logistica
    # ------------------------------------------------------------------
    print("\n1. REGRESION LOGISTICA")
    print("-" * 80)

    lr_model = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )

    cv_scores_lr = cross_val_score(lr_model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"Cross-Validation ROC-AUC: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")

    lr_model.fit(X_train, y_train)

    y_pred_lr_train = lr_model.predict(X_train)
    y_pred_lr = lr_model.predict(X_test)
    y_proba_lr = lr_model.predict_proba(X_test)[:, 1]

    print("\nMETRICAS EN TEST SET:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_lr):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_lr):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_lr):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba_lr):.4f}")
    print(f"  PR-AUC:    {average_precision_score(y_test, y_proba_lr):.4f}")

    results["Logistic Regression"] = {
        "model": lr_model,
        "y_pred_train": y_pred_lr_train,
        "y_pred": y_pred_lr,
        "y_proba": y_proba_lr,
        "cv_scores": cv_scores_lr,
    }

    # ------------------------------------------------------------------
    # 2. Arbol de Decision
    # ------------------------------------------------------------------
    print("\n2. ARBOL DE DECISION")
    print("-" * 80)

    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=random_state,
    )

    cv_scores_dt = cross_val_score(dt_model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"Cross-Validation ROC-AUC: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})")

    dt_model.fit(X_train, y_train)

    y_pred_dt_train = dt_model.predict(X_train)
    y_pred_dt = dt_model.predict(X_test)
    y_proba_dt = dt_model.predict_proba(X_test)[:, 1]

    print("\nMETRICAS EN TEST SET:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_dt):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_dt):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_dt):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba_dt):.4f}")
    print(f"  PR-AUC:    {average_precision_score(y_test, y_proba_dt):.4f}")

    results["Decision Tree"] = {
        "model": dt_model,
        "y_pred_train": y_pred_dt_train,
        "y_pred": y_pred_dt,
        "y_proba": y_proba_dt,
        "cv_scores": cv_scores_dt,
    }

    # ------------------------------------------------------------------
    # 3. Random Forest
    # ------------------------------------------------------------------
    print("\n3. RANDOM FOREST")
    print("-" * 80)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"Cross-Validation ROC-AUC: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

    rf_model.fit(X_train, y_train)

    y_pred_rf_train = rf_model.predict(X_train)
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    print("\nMETRICAS EN TEST SET:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_rf):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba_rf):.4f}")
    print(f"  PR-AUC:    {average_precision_score(y_test, y_proba_rf):.4f}")

    results["Random Forest"] = {
        "model": rf_model,
        "y_pred_train": y_pred_rf_train,
        "y_pred": y_pred_rf,
        "y_proba": y_proba_rf,
        "cv_scores": cv_scores_rf,
    }

    return results


def analyze_generalization(results: dict, y_train, y_test) -> pd.DataFrame:
    """Compara accuracy en train vs test para detectar overfitting/underfitting.

    Gap < 5%: buena generalizacion. Gap 5-10%: aceptable. Gap > 10%: posible
    overfitting.
    """
    print("\nANALISIS DE GENERALIZACION (Train vs Test)")
    print("=" * 80)

    generalization_data = []

    for model_name, model_results in results.items():
        acc_train = accuracy_score(y_train, model_results["y_pred_train"])
        acc_test = accuracy_score(y_test, model_results["y_pred"])
        gap = acc_train - acc_test

        generalization_data.append({
            "Modelo": model_name,
            "Accuracy Train": acc_train,
            "Accuracy Test": acc_test,
            "Gap (Train - Test)": gap,
            "Estado": "Overfitting" if gap > 0.10 else ("Buena Generalizacion" if gap < 0.05 else "Aceptable"),
        })

    df_gen = pd.DataFrame(generalization_data)
    print(df_gen.to_string(index=False))

    print("\nINTERPRETACION:")
    print("  - Gap < 5%: Buena generalizacion")
    print("  - Gap 5-10%: Generalizacion aceptable")
    print("  - Gap > 10%: Posible overfitting")

    return df_gen


def optimize_hyperparameters(X_train, y_train, X_test, y_test, random_state: int = RANDOM_STATE):
    """Optimiza hiperparametros de Random Forest mediante Grid Search (cv=3).

    Explora n_estimators x max_depth x min_samples_split x min_samples_leaf
    (2 x 3 x 2 x 2 = 24 combinaciones), seleccionando la de mejor ROC-AUC
    promedio en validacion cruzada.

    Returns
    -------
    tuple
        (best_model, grid_search)
    """
    print("\nOPTIMIZACION DE HIPERPARAMETROS (Grid Search)")
    print("=" * 80)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [8, 10, 12],
        "min_samples_split": [10, 20],
        "min_samples_leaf": [5, 10],
    }

    print("Grid de busqueda:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTotal de combinaciones a probar: {total_combinations}")

    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1),
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    print("\nEjecutando Grid Search (esto puede tardar un poco)...\n")
    grid_search.fit(X_train, y_train)

    print("\nMEJORES HIPERPARAMETROS ENCONTRADOS:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nMejor ROC-AUC en CV: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    y_proba_best = best_model.predict_proba(X_test)[:, 1]

    print("\nMETRICAS DEL MODELO OPTIMIZADO EN TEST:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_best):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_best):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_best):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_best):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba_best):.4f}")

    return best_model, grid_search


def compare_models(results: dict, y_test) -> pd.DataFrame:
    """Construye la tabla comparativa de metricas de todos los modelos entrenados."""
    print("\nTABLA COMPARATIVA DE MODELOS")
    print("=" * 80)

    comparison_data = []

    for model_name, model_results in results.items():
        y_pred = model_results["y_pred"]
        y_proba = model_results["y_proba"]

        comparison_data.append({
            "Modelo": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "PR-AUC": average_precision_score(y_test, y_proba),
        })

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.round(4).to_string(index=False))

    best_idx = df_comparison["F1-Score"].idxmax()
    best_model_name = df_comparison.loc[best_idx, "Modelo"]
    best_f1 = df_comparison.loc[best_idx, "F1-Score"]

    print(f"\nMEJOR MODELO: {best_model_name} (F1-Score: {best_f1:.4f})")

    return df_comparison
