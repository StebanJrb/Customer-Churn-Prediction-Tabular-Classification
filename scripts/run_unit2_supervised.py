"""
Unidad 2 - Profundizacion en Aprendizaje Supervisado
Prediccion de Churn - Caso Telmex Claro
Especializacion en Analitica de Datos - Universidad Piloto de Colombia

Evolucion de Unidad 1: agrega Arbol de Decision como tercer algoritmo,
analisis de generalizacion (overfitting/underfitting), optimizacion de
hiperparametros (Grid Search) y visualizaciones comparativas.

Run:
    python scripts/run_unit2_supervised.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from churn.config import configure_environment, set_plot_style
from churn.data import generate_telecom_dataset
from churn.preprocessing import preprocess_data
from churn.supervised_deep import (
    analyze_generalization,
    compare_models,
    optimize_hyperparameters,
    train_and_evaluate_models,
)
from churn.visualization_unit2 import create_visualizations


def main():
    configure_environment()
    set_plot_style()

    print("=" * 80)
    print("ENTRENAMIENTO DE MODELO DE APRENDIZAJE SUPERVISADO - UNIDAD 2")
    print("Prediccion de Churn - Caso Telmex Claro")
    print("=" * 80)
    print("\nCONTEXTO: Este taller profundiza en los modelos supervisados de Unidad 1")
    print("   - Unidad 1: Proyecto completo (LR + RF + K-Means)")
    print("   - Unidad 2: Profundizacion en supervisado (LR + DT + RF + Grid Search)")
    print("\nLibrerias cargadas correctamente")
    print("RANDOM_STATE = 42 para reproducibilidad\n")

    # 1. Generar datos etiquetados (mismo dataset de Unidad 1)
    df = generate_telecom_dataset(n_samples=10000)

    print("Dataset generado: {:,} clientes".format(len(df)))
    print(f"   Variables: {len(df.columns)}")
    print(f"   Etiquetas (churn_30): {df['churn_30'].sum():,} casos positivos "
          f"({df['churn_30'].mean()*100:.1f}%)")
    print(f"   Desbalance: {(1-df['churn_30'].mean())*100:.1f}% No Churn vs "
          f"{df['churn_30'].mean()*100:.1f}% Churn\n")

    print("PRIMERAS 5 FILAS DEL DATASET:")
    print(df.head())
    print("\n")

    # 2. Preprocesar datos
    X_train, X_test, y_train, y_test, features, scaler = preprocess_data(df, "churn_30")

    # 3. Entrenar y evaluar modelos (LR + Arbol de Decision + Random Forest)
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # 4. Comparar modelos
    df_comparison = compare_models(results, y_test)

    # 5. Analizar generalizacion
    df_generalization = analyze_generalization(results, y_train, y_test)

    # 6. Optimizar hiperparametros
    best_model, grid_search = optimize_hyperparameters(X_train, y_train, X_test, y_test)

    # 7. Generar visualizaciones
    create_visualizations(results, y_test, X_train)

    # ------------------------------------------------------------------
    # Resumen final y conexion con Unidad 1
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESUMEN EJECUTIVO - UNIDAD 2")
    print("=" * 80)

    print("\nCONEXION CON UNIDAD 1:")
    print("   +-------------------------------------------------------------------+")
    print("   | UNIDAD 1: Proyecto ML Completo (Formulacion)                      |")
    print("   | - Algoritmos: Regresion Logistica + Random Forest                 |")
    print("   | - Targets: churn_30 Y churn_60 (dos modelos separados)            |")
    print("   | - Clustering: K-Means con k=4 segmentos                           |")
    print("   | - Enfoque: Disenar solucion integral de negocio                   |")
    print("   +-------------------------------------------------------------------+")
    print("                              EVOLUCION")
    print("   +-------------------------------------------------------------------+")
    print("   | UNIDAD 2: Profundizacion en Supervisado (Entrenamiento)           |")
    print("   | - Algoritmos: LR + Arbol de Decision + Random Forest              |")
    print("   | - Target: Solo churn_30 (profundizar en un problema)              |")
    print("   | - Nuevas tecnicas: Grid Search, analisis generalizacion           |")
    print("   | - Enfoque: Dominar tecnicas de entrenamiento avanzadas            |")
    print("   +-------------------------------------------------------------------+")

    print("\nPROCESO COMPLETADO:")
    print("  1. [OK] Datos etiquetados generados: 10,000 clientes (MISMO dataset Unidad 1)")
    print("  2. [OK] Preprocesamiento: One-Hot Encoding + Normalizacion (StandardScaler)")
    print("  3. [OK] Particion estratificada: 80% Train (8,000) / 20% Test (2,000)")
    print("  4. [OK] Algoritmos implementados: Regresion Logistica + Arbol de Decision + Random Forest")
    print("  5. [OK] Validacion cruzada (k=5) aplicada a todos los modelos")
    print("  6. [OK] Optimizacion de hiperparametros (Grid Search exhaustivo)")
    print("  7. [OK] Visualizaciones generadas: Matrices, Curvas ROC/PR, Importancia de variables")

    best_model_name = df_comparison.loc[df_comparison["F1-Score"].idxmax(), "Modelo"]
    best_f1 = df_comparison["F1-Score"].max()
    best_roc = df_comparison.loc[df_comparison["F1-Score"].idxmax(), "ROC-AUC"]

    print(f"\nMEJOR MODELO DETECTADO: {best_model_name}")
    print(f"   |- F1-Score: {best_f1:.4f}")
    print(f"   |- ROC-AUC: {best_roc:.4f}")
    print(f"   `- Estado: {'Listo para produccion' if best_f1 > 0.70 else 'Requiere mas ajustes'}")

    print("\nANALISIS DE GENERALIZACION (Overfitting/Underfitting):")
    for _, row in df_generalization.iterrows():
        icon = "OK" if row["Estado"] == "Buena Generalizacion" else ("WARN" if row["Estado"] == "Aceptable" else "ALERT")
        print(f"   [{icon}] {row['Modelo']}: {row['Estado']} (Gap Train-Test: {row['Gap (Train - Test)']:.4f})")

    print("\nARCHIVOS GENERADOS EN CARPETA 'outputs/unit2/':")
    print("   |- matrices_confusion.png       (Comparacion de TN, FP, FN, TP)")
    print("   |- curvas_roc.png               (Discriminacion de modelos)")
    print("   |- curvas_precision_recall.png  (Trade-off Precision vs Recall)")
    print("   `- importancia_variables.png    (Top 15 features mas predictivas)")

    print("\nAPRENDIZAJES CLAVE DE UNIDAD 2:")
    print("   1. Validacion cruzada proporciona estimacion mas robusta que train/test simple")
    print("   2. Arbol de Decision captura relaciones no lineales mejor que Regresion Logistica")
    print("   3. Random Forest reduce overfitting mediante ensemble de arboles")
    print("   4. Grid Search automatiza busqueda de hiperparametros optimos")
    print("   5. Analisis de generalizacion previene deployment de modelos sobreajustados")

    print("\nPROXIMOS PASOS (Recomendaciones):")
    print("   - Probar algoritmos de Gradient Boosting (XGBoost, LightGBM, CatBoost)")
    print("   - Implementar tecnicas de balanceo de clases (SMOTE, ADASYN)")
    print("   - Realizar feature engineering (interacciones, transformaciones)")
    print("   - Calibrar probabilidades para optimizar threshold de decision basado en ROI")
    print("   - Implementar monitoreo de drift para detectar cambios en produccion")

    print("\n" + "=" * 80)
    print("UNIDAD 2 COMPLETADA EXITOSAMENTE")
    print("   Dataset: OK | Preprocesamiento: OK | Modelos: OK | Evaluacion: OK")
    print("=" * 80 + "\n")

    return {
        "models": results,
        "comparison": df_comparison,
        "generalization": df_generalization,
        "best_model": best_model,
        "data": (X_train, X_test, y_train, y_test),
    }


if __name__ == "__main__":
    output = main()
