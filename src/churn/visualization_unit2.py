"""Visualizaciones de Unidad 2: matrices de confusion, curvas ROC/PR e importancia de variables."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)

from churn.viz import output_dir

# Mapeo de nombres de modelos a espanol
_MODEL_NAMES_ES = {
    "Logistic Regression": "Regresion Logistica",
    "Decision Tree": "Arbol de Decision",
    "Random Forest": "Bosque Aleatorio",
}

# Diccionario de traduccion de variables a espanol
_VAR_NAMES_ES = {
    "tenure": "Antiguedad (meses)",
    "total_charges": "Total Facturado",
    "monthly_charges": "Cargo Mensual",
    "consumption_change": "Cambio en Consumo (%)",
    "mora_days": "Dias en Mora",
    "num_complaints": "Numero de Quejas",
    "num_tickets": "Numero de Tickets",
    "late_payments": "Pagos Atrasados",
    "month": "Mes",
    "contract_type_Month-to-month": "Contrato: Mes a Mes",
    "contract_type_One year": "Contrato: Un Ano",
    "contract_type_Two year": "Contrato: Dos Anos",
    "payment_method_Electronic check": "Pago: Cheque Electronico",
    "payment_method_Bank transfer": "Pago: Transferencia",
    "payment_method_Credit card": "Pago: Tarjeta de Credito",
    "payment_method_Mailed check": "Pago: Cheque Correo",
    "month_1": "Mes: Enero",
    "month_2": "Mes: Febrero",
    "month_3": "Mes: Marzo",
    "month_4": "Mes: Abril",
    "month_5": "Mes: Mayo",
    "month_6": "Mes: Junio",
    "month_7": "Mes: Julio",
    "month_8": "Mes: Agosto",
    "month_9": "Mes: Septiembre",
    "month_10": "Mes: Octubre",
    "month_11": "Mes: Noviembre",
    "month_12": "Mes: Diciembre",
}


def create_visualizations(results: dict, y_test, X_train) -> None:
    """Genera y guarda 4 PNGs en outputs/unit2/: matrices de confusion, curvas
    ROC, curvas precision-recall e importancia de variables (Random Forest).
    """
    print("\nGENERANDO VISUALIZACIONES...")
    print("-" * 80)

    out_dir = output_dir("unit2")

    # ------------------------------------------------------------------
    # 1. Matrices de confusion
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (model_name, model_results) in enumerate(results.items()):
        cm = confusion_matrix(y_test, model_results["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
            xticklabels=["Sin Churn", "Churn"],
            yticklabels=["Sin Churn", "Churn"],
        )
        model_name_es = _MODEL_NAMES_ES.get(model_name, model_name)
        axes[idx].set_title(f"Matriz de Confusion\n{model_name_es}", fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("Valor Real")
        axes[idx].set_xlabel("Valor Predicho")

    plt.tight_layout()
    plt.savefig(out_dir / "matrices_confusion.png", dpi=150, bbox_inches="tight")
    print("Guardado: matrices_confusion.png")
    plt.close()

    # ------------------------------------------------------------------
    # 2. Curvas ROC
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 7))

    for model_name, model_results in results.items():
        fpr, tpr, _ = roc_curve(y_test, model_results["y_proba"])
        auc = roc_auc_score(y_test, model_results["y_proba"])
        model_name_es = _MODEL_NAMES_ES.get(model_name, model_name)
        plt.plot(fpr, tpr, label=f"{model_name_es} (AUC = {auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Clasificador Aleatorio (AUC = 0.50)", linewidth=1)
    plt.xlabel("Tasa de Falsos Positivos", fontsize=12)
    plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=12)
    plt.title("Curvas ROC - Comparacion de Modelos", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "curvas_roc.png", dpi=150, bbox_inches="tight")
    print("Guardado: curvas_roc.png")
    plt.close()

    # ------------------------------------------------------------------
    # 3. Curvas Precision-Recall
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 7))

    for model_name, model_results in results.items():
        precision, recall, _ = precision_recall_curve(y_test, model_results["y_proba"])
        pr_auc = average_precision_score(y_test, model_results["y_proba"])
        model_name_es = _MODEL_NAMES_ES.get(model_name, model_name)
        plt.plot(recall, precision, label=f"{model_name_es} (PR-AUC = {pr_auc:.3f})", linewidth=2)

    plt.xlabel("Sensibilidad (Recall)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Curvas Precision-Sensibilidad - Comparacion de Modelos", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "curvas_precision_recall.png", dpi=150, bbox_inches="tight")
    print("Guardado: curvas_precision_recall.png")
    plt.close()

    # ------------------------------------------------------------------
    # 4. Importancia de variables (Random Forest)
    # ------------------------------------------------------------------
    rf_model = results["Random Forest"]["model"]
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(15)

    feature_importance["Feature_ES"] = feature_importance["Feature"].map(
        lambda x: _VAR_NAMES_ES.get(x, x)
    )

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance["Feature_ES"], feature_importance["Importance"], color="steelblue")
    plt.xlabel("Importancia", fontsize=12)
    plt.ylabel("Variable", fontsize=12)
    plt.title("Top 15 Variables Mas Importantes (Bosque Aleatorio)", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "importancia_variables.png", dpi=150, bbox_inches="tight")
    print("Guardado: importancia_variables.png")
    plt.close()

    print(f"\nTodas las visualizaciones guardadas en carpeta '{out_dir}'\n")
