"""Unidad 3: simulacion de datos faltantes (MCAR/MAR) e imputacion (Simple vs KNN)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer

from churn.viz import output_dir


def introducir_datos_faltantes(df_clientes: pd.DataFrame, num_muestras: int, semilla: int):
    """Introduce datos faltantes MCAR (5%) en varias variables y MAR en dias_mora.

    Returns
    -------
    tuple
        (df_con_faltantes, variables_con_faltantes, porcentaje_faltante)
    """
    df_con_faltantes = df_clientes.copy()

    print("\n🔧 Introduciendo datos faltantes de forma controlada...")

    np.random.seed(semilla)
    variables_con_faltantes = ['antiguedad', 'cargo_mensual', 'num_tickets', 'cambio_consumo']

    for variable in variables_con_faltantes:
        mascara_faltante = np.random.random(num_muestras) < 0.05
        df_con_faltantes.loc[mascara_faltante, variable] = np.nan
        print(f"  ✓ {variable}: {mascara_faltante.sum()} valores faltantes agregados (MCAR)")

    # MAR: Datos faltantes relacionados con otras variables
    # Clientes con muchas quejas tienen mayor probabilidad de no responder encuestas
    mascara_muchas_quejas = df_con_faltantes['num_quejas'] > 5
    probabilidad_faltante = np.where(mascara_muchas_quejas, 0.15, 0.02)
    mascara_faltante = np.random.random(num_muestras) < probabilidad_faltante
    df_con_faltantes.loc[mascara_faltante, 'dias_mora'] = np.nan
    print(f"  ✓ dias_mora: {mascara_faltante.sum()} valores faltantes agregados (MAR)")

    print("\n📊 Resumen de datos faltantes:")
    datos_faltantes = df_con_faltantes.isnull().sum()
    print(datos_faltantes[datos_faltantes > 0])
    porcentaje_faltante = df_con_faltantes.isnull().sum().sum() / (num_muestras * len(df_con_faltantes.columns)) * 100
    print(f"\n✓ Porcentaje total de valores faltantes: {porcentaje_faltante:.2f}%")

    return df_con_faltantes, variables_con_faltantes, porcentaje_faltante


def graficar_patron_faltantes(df_con_faltantes: pd.DataFrame, variables_con_faltantes: list):
    """Heatmap del patron de datos faltantes (guarda 01_patron_datos_faltantes.png)."""
    print("\n📉 Generando visualización del patrón de datos faltantes...")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_con_faltantes[variables_con_faltantes + ['dias_mora']].isnull(),
                cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Patrón de Datos Faltantes por Variable', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Variables', fontsize=12)
    plt.ylabel('Observaciones', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "01_patron_datos_faltantes.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 01_patron_datos_faltantes.png")


def comparar_metodos_imputacion(df_con_faltantes: pd.DataFrame) -> dict:
    """Compara Imputador Simple (media), KNN Imputer y eliminacion de filas (listwise).

    Returns
    -------
    dict
        df_numerico, df_imputado_simple, df_imputado_knn, df_casos_completos, filas_eliminadas
    """
    df_numerico = df_con_faltantes.select_dtypes(include=[np.number]).drop(['id_cliente'], axis=1, errors='ignore')

    print("\n🔹 3.1 Imputador Simple (estrategia: media)")
    imputador_simple = SimpleImputer(strategy='mean')
    df_imputado_simple = df_numerico.copy()
    df_imputado_simple[df_imputado_simple.columns] = imputador_simple.fit_transform(df_imputado_simple)
    print(f"  ✓ Imputación completada")
    print(f"  ✓ Datos faltantes restantes: {df_imputado_simple.isnull().sum().sum()}")

    print("\n🔹 3.2 Imputador KNN (k=5 vecinos)")
    imputador_knn = KNNImputer(n_neighbors=5)
    df_imputado_knn = df_numerico.copy()
    df_imputado_knn[df_imputado_knn.columns] = imputador_knn.fit_transform(df_imputado_knn)
    print(f"  ✓ Imputación completada")
    print(f"  ✓ Datos faltantes restantes: {df_imputado_knn.isnull().sum().sum()}")

    print("\n🔹 3.3 Eliminación de Filas (Listwise Deletion)")
    df_casos_completos = df_numerico.dropna()
    filas_eliminadas = len(df_numerico) - len(df_casos_completos)
    print(f"  ✓ Filas eliminadas: {filas_eliminadas:,}")
    print(f"  ✓ Filas restantes: {len(df_casos_completos):,} ({len(df_casos_completos)/len(df_numerico)*100:.1f}%)")

    return {
        "df_numerico": df_numerico,
        "df_imputado_simple": df_imputado_simple,
        "df_imputado_knn": df_imputado_knn,
        "df_casos_completos": df_casos_completos,
        "filas_eliminadas": filas_eliminadas,
    }


def graficar_comparacion_imputacion(df_clientes: pd.DataFrame, df_imputado_simple: pd.DataFrame, df_imputado_knn: pd.DataFrame):
    """Histogramas comparando original vs Simple vs KNN (guarda 02_comparacion_imputacion.png)."""
    print("\n📊 Generando comparación visual de métodos de imputación...")
    variables_comparacion = ['antiguedad', 'cargo_mensual', 'dias_mora']

    fig, ejes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, variable in enumerate(variables_comparacion):
        eje = ejes[idx]

        eje.hist(df_clientes[variable], bins=30, alpha=0.4, label='Original',
                color='blue', edgecolor='black', density=True)
        eje.hist(df_imputado_simple[variable], bins=30, alpha=0.4, label='Simple (Media)',
                color='green', edgecolor='black', density=True)
        eje.hist(df_imputado_knn[variable], bins=30, alpha=0.4, label='KNN (k=5)',
                color='orange', edgecolor='black', density=True)

        eje.set_title(f'{variable}', fontweight='bold', fontsize=12)
        eje.set_xlabel('Valor', fontsize=10)
        eje.set_ylabel('Densidad', fontsize=10)
        eje.legend(loc='best')
        eje.grid(True, alpha=0.3)

    plt.suptitle('Comparación de Métodos de Imputación de Datos Faltantes',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "02_comparacion_imputacion.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 02_comparacion_imputacion.png")
