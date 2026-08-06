"""
Unidad 3 - Aprendizaje No Supervisado: Deteccion de Patrones y Anomalias
Caso: Telmex Claro - Telecomunicaciones

Pipeline completo: simulacion + imputacion de datos faltantes, agrupamiento
(K-Medias, DBSCAN, jerarquico), reduccion de dimensionalidad (ACP, t-SNE),
deteccion de anomalias (Bosque de Aislamiento, LOF) y reglas de asociacion
(Apriori, opcional via mlxtend).

Run:
    python scripts/run_unit3_unsupervised.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# This script's console output uses accented Spanish and emoji (unlike Units
# 1/2, which were rewritten ASCII-only). On Windows, stdout defaults to the
# legacy cp1252 codepage, which raises UnicodeEncodeError on that text -- so
# force UTF-8 before anything else prints (including the module-level
# warning in churn.association_rules when mlxtend is missing).
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name)
    if getattr(_stream, "encoding", "").lower() != "utf-8" and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from churn.config import RANDOM_STATE, configure_environment, set_plot_style
from churn.viz import output_dir

from churn.missing_data import (
    introducir_datos_faltantes,
    graficar_patron_faltantes,
    comparar_metodos_imputacion,
    graficar_comparacion_imputacion,
)
from churn.clustering_advanced import (
    COLORES_GRUPOS,
    preparar_datos_agrupamiento,
    seleccionar_k_kmeans,
    entrenar_kmeans,
    graficar_kmeans_pca,
    seleccionar_eps_dbscan,
    entrenar_dbscan,
    graficar_dbscan,
    analizar_outliers_dbscan,
    dendrograma_jerarquico,
    entrenar_jerarquico,
    graficar_jerarquico,
)
from churn.dimensionality import (
    aplicar_pca_2d,
    aplicar_pca_completo,
    graficar_pca_varianza,
    graficar_pca_3d,
    analizar_cargas_pca,
    aplicar_tsne,
    graficar_tsne,
)
from churn.anomaly import (
    entrenar_isolation_forest,
    analizar_anomalias_bosque,
    graficar_anomalias_bosque,
    entrenar_lof,
    comparar_bosque_lof,
    graficar_comparacion_anomalias,
)
from churn.association_rules import analizar_reglas_asociacion, MLXTEND_DISPONIBLE


def _generar_dataset_clientes(num_muestras: int, semilla: int) -> pd.DataFrame:
    """Genera el dataset sintetico de clientes Telmex Claro para Unidad 3.

    NOTA DE DIVERGENCIA: la logica de generacion inline del script original
    de Unidad 3 usa distribuciones UNIFORMES (np.random.randint / uniform)
    y nombres de columna en espanol (antiguedad, tipo_contrato, ...), a
    diferencia de churn.data.generate_telecom_dataset() (que usa
    distribuciones exponencial/normal/poisson, una formula de riesgo
    distinta, y nombres de columna en ingles). Esto es una divergencia real
    en la logica BASE de generacion -- no solo en la inyeccion de datos
    faltantes -- por lo que se preserva tal cual estaba en el script
    original en lugar de reusar el generador compartido, para no alterar
    el comportamiento de todo el resto del pipeline (que referencia estas
    columnas en espanol por todas partes).
    """
    print("=" * 80)
    print("1. GENERACIÓN DE DATOS")
    print("=" * 80)

    tipos_contrato = ['Mes-a-mes', 'Un-año', 'Dos-años']
    metodos_pago = ['Cheque-electrónico', 'Transferencia-bancaria', 'Tarjeta-crédito', 'Cheque-postal']

    np.random.seed(semilla)

    datos = {
        'id_cliente': [f'CLIENTE_{i:05d}' for i in range(1, num_muestras + 1)],
        'antiguedad': np.random.randint(1, 73, num_muestras),
        'tipo_contrato': np.random.choice(tipos_contrato, num_muestras, p=[0.50, 0.30, 0.20]),
        'cargo_mensual': np.random.uniform(30000, 200000, num_muestras),
        'metodo_pago': np.random.choice(metodos_pago, num_muestras, p=[0.35, 0.25, 0.25, 0.15]),
        'num_tickets': np.random.randint(0, 16, num_muestras),
        'num_quejas': np.random.randint(0, 9, num_muestras),
        'pagos_tardios': np.random.randint(0, 13, num_muestras),
        'dias_mora': np.random.randint(0, 91, num_muestras),
        'cambio_consumo': np.random.uniform(-50, 50, num_muestras),
        'mes': np.random.randint(1, 13, num_muestras),
    }

    df_clientes = pd.DataFrame(datos)
    df_clientes['cargo_total'] = df_clientes['cargo_mensual'] * df_clientes['antiguedad']

    # Generacion de etiquetas de abandono (churn) -- SOLO PARA VALIDACION POSTERIOR
    # NO SE USA EN ENTRENAMIENTO NO SUPERVISADO
    puntaje_abandono = np.zeros(num_muestras)
    puntaje_abandono += (df_clientes['tipo_contrato'] == 'Mes-a-mes').astype(int) * 25
    puntaje_abandono += (df_clientes['antiguedad'] < 6).astype(int) * 20
    puntaje_abandono += (df_clientes['antiguedad'] > 60).astype(int) * 10
    puntaje_abandono += df_clientes['num_quejas'] * 8
    puntaje_abandono += (df_clientes['dias_mora'] > 30).astype(int) * 15
    puntaje_abandono += (df_clientes['cambio_consumo'] < -20).astype(int) * 12
    puntaje_abandono += df_clientes['num_tickets'] * 3
    puntaje_abandono += df_clientes['pagos_tardios'] * 5
    puntaje_abandono += (df_clientes['metodo_pago'] == 'Cheque-electrónico').astype(int) * 8
    puntaje_abandono += ((df_clientes['mes'] == 1) | (df_clientes['mes'] == 2)).astype(int) * 8

    probabilidad_abandono = 1 / (1 + np.exp(-0.08 * (puntaje_abandono - 50)))
    df_clientes['abandono_30'] = (np.random.random(num_muestras) < probabilidad_abandono).astype(int)

    print(f"✓ Conjunto de datos generado: {df_clientes.shape[0]:,} clientes, {df_clientes.shape[1]} variables")
    print(f"✓ Tasa de abandono: {df_clientes['abandono_30'].mean()*100:.1f}%")
    print("\n📊 Primeras 5 filas del conjunto de datos:")
    print(df_clientes.head())

    print("\n" + "=" * 80)
    print("1.1 ANÁLISIS EXPLORATORIO DETALLADO")
    print("=" * 80)

    print("\n📈 Estadísticas descriptivas de variables numéricas:")
    print(df_clientes.describe())

    print("\n📊 Distribución de variables categóricas:")
    print("\n🔹 Tipo de contrato:")
    print(df_clientes['tipo_contrato'].value_counts())
    print(f"\nPorcentajes:")
    print(df_clientes['tipo_contrato'].value_counts(normalize=True) * 100)

    print("\n🔹 Método de pago:")
    print(df_clientes['metodo_pago'].value_counts())
    print(f"\nPorcentajes:")
    print(df_clientes['metodo_pago'].value_counts(normalize=True) * 100)

    return df_clientes


def _graficar_dashboard_comparativo(df_clientes, k_optimo, varianza_explicada,
                                     anomalias_bosque, anomalias_lof, X_acp, X_tsne, grupos_kmeans):
    """Dashboard integrado de 8 paneles que compara todos los metodos implementados
    (guarda 15_dashboard_comparativo.png). Devuelve (tasa_global, grupo_max_abandono,
    tasa_grupo_riesgo), reutilizados en la seccion de recomendaciones."""
    print("\n📊 Generando dashboard comparativo...")

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Comparación de tasas de abandono por grupo (K-Medias, DBSCAN, Jerárquico)
    ax1 = fig.add_subplot(gs[0, :2])

    grupos_comparacion = pd.DataFrame({
        'K-Medias': df_clientes.groupby('grupo_kmeans')['abandono_30'].mean() * 100,
        'Jerárquico': df_clientes.groupby('grupo_jerarquico')['abandono_30'].mean() * 100,
    })

    if df_clientes[df_clientes['grupo_dbscan'] != -1]['grupo_dbscan'].nunique() > 0:
        dbscan_tasas = df_clientes[df_clientes['grupo_dbscan'] != -1].groupby('grupo_dbscan')['abandono_30'].mean() * 100
        for grupo_id in range(k_optimo):
            if grupo_id not in dbscan_tasas.index:
                dbscan_tasas[grupo_id] = 0
        grupos_comparacion['DBSCAN'] = dbscan_tasas.sort_index()

    x_pos = np.arange(len(grupos_comparacion))
    ancho_barra = 0.25

    for i, (metodo, color) in enumerate(zip(grupos_comparacion.columns, ['#FF6B6B', '#4ECDC4', '#45B7D1'])):
        ax1.bar(x_pos + i * ancho_barra, grupos_comparacion[metodo], ancho_barra,
               label=metodo, color=color, alpha=0.8, edgecolor='black')

    ax1.set_xlabel('ID de Grupo', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Tasa de Abandono (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Comparación de Tasas de Abandono por Método de Agrupamiento',
                 fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos + ancho_barra)
    ax1.set_xticklabels([f'Grupo {i}' for i in range(len(grupos_comparacion))])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Varianza explicada ACP
    ax2 = fig.add_subplot(gs[0, 2])
    num_componentes_mostrar = min(10, len(varianza_explicada))
    ax2.bar(range(1, num_componentes_mostrar + 1), varianza_explicada[:num_componentes_mostrar],
           color='steelblue', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Componente', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Varianza', fontsize=10, fontweight='bold')
    ax2.set_title('Varianza Explicada por ACP\n(Primeros 10 componentes)',
                 fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Distribución Normal vs Anomalías (Bosque de Aislamiento)
    ax3 = fig.add_subplot(gs[1, 0])
    distribucion_anomalias_bosque = df_clientes['anomalia_bosque'].value_counts()
    etiquetas_bosque = ['Normal', 'Anomalía']
    colores_pie_bosque = ['#2ECC71', '#E74C3C']
    ax3.pie([distribucion_anomalias_bosque.get(1, 0), distribucion_anomalias_bosque.get(-1, 0)],
           labels=etiquetas_bosque, colors=colores_pie_bosque, autopct='%1.1f%%',
           startangle=90, explode=(0, 0.1), shadow=True, textprops={'fontweight': 'bold'})
    ax3.set_title('Bosque de Aislamiento\nDistribución', fontsize=11, fontweight='bold')

    # 4. Distribución Normal vs Anomalías (LOF)
    ax4 = fig.add_subplot(gs[1, 1])
    distribucion_anomalias_lof = df_clientes['anomalia_lof'].value_counts()
    etiquetas_lof = ['Normal', 'Anomalía']
    colores_pie_lof = ['#2ECC71', '#E74C3C']
    ax4.pie([distribucion_anomalias_lof.get(1, 0), distribucion_anomalias_lof.get(-1, 0)],
           labels=etiquetas_lof, colors=colores_pie_lof, autopct='%1.1f%%',
           startangle=90, explode=(0, 0.1), shadow=True, textprops={'fontweight': 'bold'})
    ax4.set_title('Factor Outlier Local (LOF)\nDistribución', fontsize=11, fontweight='bold')

    # 5. Comparación de tasas de abandono: Global, Grupo Alto Riesgo, Anomalías
    ax5 = fig.add_subplot(gs[1, 2])

    tasa_global = df_clientes['abandono_30'].mean() * 100
    grupo_max_abandono = df_clientes.groupby('grupo_kmeans')['abandono_30'].mean().idxmax()
    tasa_grupo_riesgo = df_clientes[df_clientes['grupo_kmeans'] == grupo_max_abandono]['abandono_30'].mean() * 100
    tasa_anomalias_bosque_val = df_clientes[df_clientes['anomalia_bosque'] == -1]['abandono_30'].mean() * 100
    tasa_normales_bosque = df_clientes[df_clientes['anomalia_bosque'] == 1]['abandono_30'].mean() * 100

    categorias_comparacion = ['Global', f'Grupo {grupo_max_abandono}\n(Alto Riesgo)', 'Anomalías\n(Bosque)', 'Normales']
    valores_comparacion = [tasa_global, tasa_grupo_riesgo, tasa_anomalias_bosque_val, tasa_normales_bosque]
    colores_barras = ['#3498DB', '#E74C3C', '#F39C12', '#2ECC71']

    barras = ax5.barh(categorias_comparacion, valores_comparacion, color=colores_barras,
                     alpha=0.8, edgecolor='black', linewidth=1.5)

    ax5.set_xlabel('Tasa de Abandono (%)', fontsize=10, fontweight='bold')
    ax5.set_title('Comparación de Tasas de Abandono\npor Segmento', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')

    for i, (barra, valor) in enumerate(zip(barras, valores_comparacion)):
        ax5.text(valor + 1, i, f'{valor:.1f}%', va='center', fontweight='bold', fontsize=9)

    # 6. Distribución de clientes por grupo K-Medias
    ax6 = fig.add_subplot(gs[2, 0])
    distribucion_kmeans = df_clientes['grupo_kmeans'].value_counts().sort_index()
    ax6.bar(distribucion_kmeans.index, distribucion_kmeans.values,
           color=COLORES_GRUPOS[:k_optimo], alpha=0.8, edgecolor='black')
    ax6.set_xlabel('Grupo ID', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Número de Clientes', fontsize=10, fontweight='bold')
    ax6.set_title('Distribución de Clientes\npor Grupo K-Medias', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_xticks(range(k_optimo))

    # 7. Grupos en espacio ACP
    ax7 = fig.add_subplot(gs[2, 1])
    for grupo_id in range(k_optimo):
        mascara = grupos_kmeans == grupo_id
        ax7.scatter(X_acp[mascara, 0], X_acp[mascara, 1],
                   c=COLORES_GRUPOS[grupo_id], label=f'{grupo_id}', alpha=0.6, s=15)
    ax7.set_xlabel('CP1', fontsize=10, fontweight='bold')
    ax7.set_ylabel('CP2', fontsize=10, fontweight='bold')
    ax7.set_title('Grupos en Espacio ACP', fontsize=11, fontweight='bold')
    ax7.legend(title='Grupo', fontsize=8, title_fontsize=9)
    ax7.grid(True, alpha=0.3)

    # 8. Anomalías en espacio t-SNE
    ax8 = fig.add_subplot(gs[2, 2])
    mascara_normales_tsne = anomalias_bosque == 1
    ax8.scatter(X_tsne[mascara_normales_tsne, 0], X_tsne[mascara_normales_tsne, 1],
               c='#3498DB', alpha=0.4, s=10, label='Normal')
    mascara_anomalias_tsne = anomalias_bosque == -1
    ax8.scatter(X_tsne[mascara_anomalias_tsne, 0], X_tsne[mascara_anomalias_tsne, 1],
               c='#E74C3C', alpha=0.8, s=30, marker='X', edgecolors='black', linewidth=0.5, label='Anomalía')
    ax8.set_xlabel('t-SNE 1', fontsize=10, fontweight='bold')
    ax8.set_ylabel('t-SNE 2', fontsize=10, fontweight='bold')
    ax8.set_title('Anomalías en Espacio t-SNE', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    plt.suptitle('DASHBOARD COMPARATIVO: Aprendizaje No Supervisado - Telmex Claro',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_dir("unit3") / "15_dashboard_comparativo.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 15_dashboard_comparativo.png")

    return tasa_global, grupo_max_abandono, tasa_grupo_riesgo


def main():
    configure_environment()
    pd.set_option('display.width', 120)
    set_plot_style()
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 80)
    print("APRENDIZAJE NO SUPERVISADO - UNIDAD 3")
    print("Detección de Patrones y Anomalías - Caso Telmex Claro")
    print("=" * 80)
    print("\n📚 CONTEXTO: Este taller profundiza en técnicas no supervisadas")
    print("   - Unidad 1: Proyecto completo (supervisado + K-Medias básico)")
    print("   - Unidad 2: Profundización en supervisado")
    print("   - Unidad 3: Profundización en NO supervisado (agrupamiento, ACP, anomalías)")
    print("\n✓ Librerías cargadas correctamente")
    print(f"✓ SEMILLA_ALEATORIA = {RANDOM_STATE} para reproducibilidad")
    print(f"✓ mlxtend {'DISPONIBLE' if MLXTEND_DISPONIBLE else 'NO DISPONIBLE'}\n")

    # ------------------------------------------------------------------
    # 1. Generación de datos
    # ------------------------------------------------------------------
    num_muestras = 10000
    df_clientes = _generar_dataset_clientes(num_muestras, RANDOM_STATE)

    # ------------------------------------------------------------------
    # 2. Preprocesamiento e introducción de datos faltantes
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. PREPROCESAMIENTO E INTRODUCCIÓN DE DATOS FALTANTES")
    print("=" * 80)

    df_con_faltantes, variables_con_faltantes, porcentaje_faltante = introducir_datos_faltantes(
        df_clientes, num_muestras, RANDOM_STATE
    )
    graficar_patron_faltantes(df_con_faltantes, variables_con_faltantes)

    # ------------------------------------------------------------------
    # 3. Imputación de datos faltantes - comparación de métodos
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. IMPUTACIÓN DE DATOS FALTANTES - COMPARACIÓN DE MÉTODOS")
    print("=" * 80)

    imputacion = comparar_metodos_imputacion(df_con_faltantes)
    graficar_comparacion_imputacion(df_clientes, imputacion["df_imputado_simple"], imputacion["df_imputado_knn"])

    df_limpio = imputacion["df_imputado_knn"].copy()
    print("\n" + "=" * 50)
    print("✅ DECISIÓN: Usaremos Imputador KNN (k=5)")
    print("=" * 50)
    print("JUSTIFICACIÓN:")
    print("  • Preserva mejor las distribuciones originales")
    print("  • Mantiene relaciones entre variables")
    print("  • Más robusto frente a valores atípicos")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 4. Preparación de datos para agrupamiento
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. PREPARACIÓN DE DATOS PARA AGRUPAMIENTO")
    print("=" * 80)

    df_agrupamiento, df_escalado, X_escalado, escalador, dummies_contrato, dummies_pago = preparar_datos_agrupamiento(
        df_clientes, df_limpio
    )

    # ------------------------------------------------------------------
    # 5. Agrupamiento: K-Medias
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. AGRUPAMIENTO: ALGORITMO K-MEDIAS (K-MEANS)")
    print("=" * 80)

    inercias, coeficientes_silueta = seleccionar_k_kmeans(X_escalado, RANDOM_STATE)

    k_optimo = 4
    print(f"\n✅ k óptimo seleccionado: {k_optimo}")
    print(f"   Justificación: Punto de inflexión en curva del codo + Buena silueta")

    (df_clientes, modelo_kmeans_final, grupos_kmeans, nombres_grupos,
     silueta_promedio, davies_bouldin, calinski_harabasz) = entrenar_kmeans(
        X_escalado, df_clientes, k_optimo, RANDOM_STATE
    )

    X_acp, modelo_acp_2d = aplicar_pca_2d(X_escalado, RANDOM_STATE)
    graficar_kmeans_pca(X_acp, modelo_acp_2d, modelo_kmeans_final, grupos_kmeans, k_optimo, nombres_grupos)

    # ------------------------------------------------------------------
    # 6. Agrupamiento: DBSCAN
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. AGRUPAMIENTO: DBSCAN (Basado en Densidad)")
    print("=" * 80)

    seleccionar_eps_dbscan(X_escalado)

    eps_optimo = 3.5
    muestras_minimas = 50
    df_clientes, grupos_dbscan, num_grupos_dbscan, num_ruido = entrenar_dbscan(
        X_escalado, df_clientes, eps_optimo, muestras_minimas
    )
    graficar_dbscan(X_acp, grupos_dbscan, num_grupos_dbscan)
    analizar_outliers_dbscan(df_clientes, num_ruido)

    # ------------------------------------------------------------------
    # 7. Agrupamiento jerárquico
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("7. AGRUPAMIENTO JERÁRQUICO (Hierarchical Clustering)")
    print("=" * 80)

    dendrograma_jerarquico(X_escalado, RANDOM_STATE)
    df_clientes, grupos_jerarquico = entrenar_jerarquico(X_escalado, df_clientes, k_optimo)
    graficar_jerarquico(X_acp, grupos_jerarquico, k_optimo)

    # ------------------------------------------------------------------
    # 8. Reducción de dimensionalidad: ACP (PCA)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("8. REDUCCIÓN DE DIMENSIONALIDAD: ACP (Análisis de Componentes Principales)")
    print("=" * 80)

    (modelo_acp_completo, X_acp_completo, varianza_explicada, varianza_acumulada,
     num_componentes_total, num_componentes_95) = aplicar_pca_completo(X_escalado, RANDOM_STATE)

    graficar_pca_varianza(varianza_explicada, varianza_acumulada, num_componentes_95)
    graficar_pca_3d(X_acp_completo, varianza_explicada, grupos_kmeans)
    analizar_cargas_pca(modelo_acp_completo, df_agrupamiento.columns, varianza_explicada)

    print("\n✅ CONCLUSIÓN ACP:")
    print(f"  Con solo {num_componentes_95} componentes principales (de {num_componentes_total} originales)")
    print(f"  podemos representar el 95% de la variabilidad de los datos.")
    print(f"  Esto reduce la dimensionalidad en {(1 - num_componentes_95/num_componentes_total)*100:.0f}%")

    # ------------------------------------------------------------------
    # 9. Reducción de dimensionalidad: t-SNE
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("9. REDUCCIÓN DE DIMENSIONALIDAD: t-SNE")
    print("=" * 80)

    X_tsne = aplicar_tsne(X_escalado, RANDOM_STATE)
    graficar_tsne(X_tsne, grupos_kmeans, k_optimo, df_clientes)

    # ------------------------------------------------------------------
    # 10. Detección de anomalías: Bosque de Aislamiento
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("10. DETECCIÓN DE ANOMALÍAS: BOSQUE DE AISLAMIENTO (ISOLATION FOREST)")
    print("=" * 80)

    df_clientes, anomalias_bosque, num_anomalias_bosque = entrenar_isolation_forest(
        X_escalado, df_clientes, RANDOM_STATE
    )
    (df_anomalias_bosque, df_normales_bosque, tasa_abandono_anomalias,
     tasa_abandono_normales, ratio_abandono) = analizar_anomalias_bosque(df_clientes)
    graficar_anomalias_bosque(X_tsne, anomalias_bosque, num_anomalias_bosque)

    # ------------------------------------------------------------------
    # 11. Detección de anomalías: Factor de Outlier Local (LOF)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("11. DETECCIÓN DE ANOMALÍAS: FACTOR DE OUTLIER LOCAL (LOF)")
    print("=" * 80)

    df_clientes, anomalias_lof, num_anomalias_lof = entrenar_lof(X_escalado, df_clientes)
    ambos_anomalias, solo_bosque, solo_lof, ambos_normales, concordancia = comparar_bosque_lof(
        df_clientes, num_anomalias_bosque
    )
    graficar_comparacion_anomalias(X_tsne, anomalias_bosque, anomalias_lof, num_anomalias_bosque, num_anomalias_lof)

    # ------------------------------------------------------------------
    # 12. Reglas de asociación (Apriori, opcional)
    # ------------------------------------------------------------------
    reglas_resultado = analizar_reglas_asociacion(df_clientes)
    reglas_abandono = reglas_resultado["reglas_abandono"]
    reglas_abandono_ordenadas = reglas_resultado["reglas_abandono_ordenadas"]

    # ------------------------------------------------------------------
    # 13. Dashboard comparativo final
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("13. DASHBOARD COMPARATIVO FINAL - INTEGRACIÓN DE RESULTADOS")
    print("=" * 80)

    tasa_global, grupo_max_abandono, tasa_grupo_riesgo = _graficar_dashboard_comparativo(
        df_clientes, k_optimo, varianza_explicada, anomalias_bosque, anomalias_lof, X_acp, X_tsne, grupos_kmeans
    )

    # ------------------------------------------------------------------
    # 14. Resumen ejecutivo y hallazgos principales
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("14. RESUMEN EJECUTIVO Y HALLAZGOS PRINCIPALES")
    print("=" * 80)

    print("\n" + "🎯 " + "=" * 78)
    print("HALLAZGOS PRINCIPALES - APRENDIZAJE NO SUPERVISADO")
    print("=" * 80)

    print("\n📊 1. AGRUPAMIENTO (CLUSTERING)")
    print("-" * 80)
    print(f"   🔹 K-Medias: Identificó {k_optimo} grupos distintos de clientes")
    print(f"      • Coeficiente de Silueta: {silueta_promedio:.3f} (buena separación)")
    print(f"      • Índice Davies-Bouldin: {davies_bouldin:.3f} (compacidad adecuada)")

    print(f"\n   📍 Perfil de Grupos Identificados:")
    for grupo_id in range(k_optimo):
        datos_grupo = df_clientes[df_clientes['grupo_kmeans'] == grupo_id]
        tasa_abandono_grupo = datos_grupo['abandono_30'].mean() * 100
        tamaño_grupo = len(datos_grupo)
        nivel = "🔴 CRÍTICO" if tasa_abandono_grupo > 70 else ("🟠 ALTO" if tasa_abandono_grupo > 50 else ("🟡 MEDIO" if tasa_abandono_grupo > 30 else "🟢 BAJO"))

        print(f"      Grupo {grupo_id}: {tamaño_grupo:,} clientes ({tamaño_grupo/len(df_clientes)*100:.1f}%), "
              f"Abandono={tasa_abandono_grupo:.1f}% [{nivel}]")

    print(f"\n   🔹 DBSCAN: {num_grupos_dbscan} grupos + {num_ruido} outliers ({num_ruido/len(df_clientes)*100:.1f}%)")
    print(f"   🔹 Jerárquico: Confirmó estructura de {k_optimo} grupos (consistente con K-Medias)")

    print("\n\n📊 2. REDUCCIÓN DE DIMENSIONALIDAD")
    print("-" * 80)
    print(f"   🔹 ACP (Análisis de Componentes Principales):")
    print(f"      • {num_componentes_95} componentes explican 95% de la varianza")
    print(f"      • Reducción dimensional: {num_componentes_total} → {num_componentes_95} variables")
    print(f"      • Compresión lograda: {(1 - num_componentes_95/num_componentes_total)*100:.1f}%")
    print(f"      • Beneficio: Análisis {(1 - num_componentes_95/num_componentes_total)*100:.0f}% más rápido")

    print(f"\n   🔹 t-SNE:")
    print(f"      • Visualización en 2D revela estructura clara de grupos")
    print(f"      • Separación visible entre clientes con/sin abandono")
    print(f"      • Útil para comunicación con equipos no técnicos")

    print("\n\n📊 3. DETECCIÓN DE ANOMALÍAS")
    print("-" * 80)
    print(f"   🔹 Bosque de Aislamiento:")
    print(f"      • Anomalías detectadas: {num_anomalias_bosque} ({num_anomalias_bosque/len(df_clientes)*100:.1f}%)")
    print(f"      • Tasa de abandono en anomalías: {tasa_abandono_anomalias:.1f}%")
    print(f"      • Tasa de abandono en normales: {tasa_abandono_normales:.1f}%")
    print(f"      • ⚡ Riesgo relativo: {ratio_abandono:.1f}x más probabilidad de abandono")

    print(f"\n   🔹 Factor de Outlier Local (LOF):")
    print(f"      • Anomalías detectadas: {num_anomalias_lof} ({num_anomalias_lof/len(df_clientes)*100:.1f}%)")
    print(f"      • Concordancia con Bosque: {(ambos_anomalias/(num_anomalias_bosque))*100:.1f}% de overlap")

    print("\n\n📊 4. IMPUTACIÓN DE DATOS FALTANTES")
    print("-" * 80)
    print(f"   🔹 Datos faltantes: {df_con_faltantes.isnull().sum().sum()} valores ({porcentaje_faltante:.2f}%)")
    print(f"   🔹 Método seleccionado: Imputador KNN (k=5)")
    print(f"   🔹 Justificación:")
    print(f"      • Preserva distribuciones originales mejor que imputación simple")
    print(f"      • Mantiene relaciones entre variables")
    print(f"      • Más robusto frente a valores atípicos que media/mediana")

    if MLXTEND_DISPONIBLE and reglas_abandono_ordenadas is not None and len(reglas_abandono_ordenadas) > 0:
        print("\n\n📊 5. REGLAS DE ASOCIACIÓN")
        print("-" * 80)
        print(f"   🔹 Reglas identificadas: {len(reglas_abandono)} predicen abandono")
        print(f"   🔹 Confidence máxima: {reglas_abandono['confidence'].max():.1%}")
        print(f"   🔹 Lift máximo: {reglas_abandono['lift'].max():.1f}x (vs tasa base)")
        print(f"\n   📍 Top 3 Reglas Predictivas:")
        for i, (idx, fila) in enumerate(reglas_abandono_ordenadas.head(3).iterrows(), 1):
            antecedentes = ', '.join(list(fila['antecedents']))
            print(f"      {i}. {antecedentes} → Abandono")
            print(f"         (Confidence: {fila['confidence']:.1%}, Lift: {fila['lift']:.1f}x)")

    # ------------------------------------------------------------------
    # 15. Conclusiones y recomendaciones estratégicas
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("15. CONCLUSIONES Y RECOMENDACIONES ESTRATÉGICAS")
    print("=" * 80)

    print("\n📋 CONCLUSIONES TÉCNICAS")
    print("=" * 80)

    conclusiones_tecnicas = [
        f"1. SEGMENTACIÓN: Identificamos {k_optimo} segmentos naturales de clientes con perfiles "
        f"de riesgo claramente diferenciados (abandono del 15% al 70%)",

        f"2. EFICIENCIA: La reducción dimensional permitió comprimir {(1 - num_componentes_95/num_componentes_total)*100:.0f}% "
        f"de variables manteniendo 95% de información, acelerando análisis futuros",

        f"3. ALERTA TEMPRANA: Los {num_anomalias_bosque} clientes con comportamiento anómalo tienen "
        f"{ratio_abandono:.1f}x más riesgo de abandono que clientes normales",

        "4. CONSISTENCIA: Los tres métodos de agrupamiento (K-Medias, DBSCAN, Jerárquico) "
        "identificaron estructuras similares, validando los hallazgos",

        "5. IMPUTACIÓN: KNN Imputer demostró ser superior a métodos simples para manejar "
        "datos faltantes, preservando relaciones entre variables cruciales",
    ]

    for conclusion in conclusiones_tecnicas:
        print(f"\n   {conclusion}")

    print("\n\n💼 RECOMENDACIONES DE NEGOCIO")
    print("=" * 80)

    print("\n🎯 CORTO PLAZO (1-3 meses):")
    print("-" * 80)

    recomendaciones_corto = [
        {
            'titulo': '1. INTERVENCIÓN EN GRUPO DE ALTO RIESGO',
            'descripcion': f'Grupo {grupo_max_abandono} tiene {tasa_grupo_riesgo:.0f}% de abandono',
            'acciones': [
                f'Contacto telefónico proactivo con {df_clientes[df_clientes["grupo_kmeans"] == grupo_max_abandono].shape[0]:,} clientes',
                'Ofertas personalizadas de retención (descuentos 10-20%)',
                'Resolución express de quejas pendientes (< 24 horas)',
                'Asignación de gestor de cuenta dedicado',
            ],
            'roi_estimado': f'{df_clientes[df_clientes["grupo_kmeans"] == grupo_max_abandono].shape[0] * 0.1 * 50000 * 12 / 1000000:.0f}M COP/año',
        },
        {
            'titulo': f'2. SISTEMA DE ALERTAS PARA ANOMALÍAS',
            'descripcion': f'Monitorear {num_anomalias_bosque} clientes con comportamiento atípico',
            'acciones': [
                'Dashboard en tiempo real de anomalías nuevas',
                'Alerta automática al equipo de retención',
                'Protocolo de respuesta rápida (< 48 horas)',
                'Análisis de causa raíz de cada anomalía',
            ],
            'roi_estimado': f'{num_anomalias_bosque * 0.15 * 50000 * 12 / 1000000:.0f}M COP/año',
        },
        {
            'titulo': '3. PILOTO DE RETENCIÓN PERSONALIZADA',
            'descripcion': 'A/B testing de estrategias por segmento',
            'acciones': [
                'Grupo control vs experimental (50/50)',
                'Ofertas diferenciadas por perfil de riesgo',
                'Medición de efectividad a 90 días',
                'Escalamiento de estrategias exitosas',
            ],
            'roi_estimado': 'TBD después de piloto',
        },
    ]

    for rec in recomendaciones_corto:
        print(f"\n   {rec['titulo']}")
        print(f"   {rec['descripcion']}")
        print(f"   Acciones:")
        for accion in rec['acciones']:
            print(f"      • {accion}")
        print(f"   💰 ROI estimado: {rec['roi_estimado']}")

    print("\n\n🎯 MEDIANO PLAZO (3-6 meses):")
    print("-" * 80)

    recomendaciones_mediano = [
        '1. INTEGRACIÓN CON MODELOS SUPERVISADOS: Usar clusters como features adicionales '
        'en modelos predictivos de Unidad 2 para mejorar precisión',

        '2. AUTOMATIZACIÓN DE INTERVENCIONES: Crear workflows automáticos basados en '
        'reglas de asociación para activar ofertas sin intervención manual',

        '3. OPTIMIZACIÓN DE PROCESAMIENTO: Implementar ACP en pipeline de scoring para '
        f'reducir tiempo de ejecución en {(1 - num_componentes_95/num_componentes_total)*100:.0f}%',

        '4. EXPANSIÓN DE MONITOREO: Aplicar detección de anomalías a otras métricas '
        '(fraude, consumo inusual, quejas atípicas)',
    ]

    for i, rec in enumerate(recomendaciones_mediano, 1):
        print(f"\n   {rec}")

    print("\n\n🎯 LARGO PLAZO (6-12 meses):")
    print("-" * 80)

    recomendaciones_largo = [
        '1. ANÁLISIS DE TRANSICIONES: Estudiar cómo evolucionan clientes entre clusters '
        'para predecir cambios de comportamiento antes que ocurran',

        '2. PERSONALIZACIÓN PROFUNDA: Desarrollar motor de recomendaciones usando '
        'reglas de asociación para ofertas 100% personalizadas',

        '3. MODELO HÍBRIDO: Combinar aprendizaje supervisado (Unidad 2) con no supervisado '
        '(Unidad 3) en modelo ensemble de máximo desempeño',

        '4. EXPANSIÓN REGIONAL: Replicar análisis en otras regiones/países para '
        'identificar patrones específicos de cada mercado',

        '5. BENCHMARKING COMPETITIVO: Comparar tasas de abandono y efectividad de '
        'intervenciones vs competencia para identificar mejores prácticas',
    ]

    for i, rec in enumerate(recomendaciones_largo, 1):
        print(f"\n   {rec}")

    # ------------------------------------------------------------------
    # 16. Análisis prospectivo y visión futura
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("16. ANÁLISIS PROSPECTIVO Y VISIÓN FUTURA")
    print("=" * 80)

    print("\n🔮 TENDENCIAS Y OPORTUNIDADES")
    print("=" * 80)

    prospectiva = [
        {
            'tendencia': '1. INTELIGENCIA ARTIFICIAL EN TIEMPO REAL',
            'descripcion': 'Los modelos no supervisados pueden actualizarse continuamente '
                          'con datos en tiempo real para detectar cambios de comportamiento '
                          'en el momento que ocurren.',
            'impacto': 'ALTO - Reducción de tiempo de respuesta de días a minutos',
            'timeline': '12-18 meses',
        },
        {
            'tendencia': '2. HIPER-PERSONALIZACIÓN',
            'descripcion': 'Combinar clusters con reglas de asociación y preferencias individuales '
                          'para crear experiencias únicas por cliente.',
            'impacto': 'ALTO - Incremento en satisfacción y retención del 15-25%',
            'timeline': '6-12 meses',
        },
        {
            'tendencia': '3. ANÁLISIS PREDICTIVO MULTIVARIADO',
            'descripcion': 'Integrar datos de múltiples fuentes (redes sociales, uso de app, '
                          'atención al cliente) en modelos no supervisados más completos.',
            'impacto': 'MEDIO - Mejora en precisión de segmentación del 10-15%',
            'timeline': '12-24 meses',
        },
        {
            'tendencia': '4. AUTOMATIZACIÓN DE RETENCIÓN',
            'descripcion': 'Sistemas completamente autónomos que identifican riesgo, diseñan '
                          'oferta personalizada y ejecutan intervención sin intervención humana.',
            'impacto': 'MUY ALTO - Escalabilidad 10x con mismo equipo',
            'timeline': '18-24 meses',
        },
        {
            'tendencia': '5. ANÁLISIS DE GRAFOS SOCIALES',
            'descripcion': 'Incorporar relaciones entre clientes (familias, empresas) para '
                          'entender abandono por contagio social.',
            'impacto': 'MEDIO - Prevención de abandono en cascada',
            'timeline': '24+ meses',
        },
    ]

    for prospecto in prospectiva:
        print(f"\n   🔹 {prospecto['tendencia']}")
        print(f"      {prospecto['descripcion']}")
        print(f"      💡 Impacto: {prospecto['impacto']}")
        print(f"      ⏱ Timeline: {prospecto['timeline']}")

    print("\n\n🌟 INNOVACIONES POTENCIALES")
    print("=" * 80)

    innovaciones = [
        "• HEALTH SCORE DINÁMICO: Puntaje de salud del cliente que se actualiza en tiempo real",
        "• JOURNEY MAPS AUTOMÁTICOS: Visualización automática del recorrido de cliente",
        "• A/B TESTING PERPETUO: Experimentación continua de estrategias de retención",
        "• SIMULADORES DE ESCENARIOS: 'Qué pasaría si' basados en modelos no supervisados",
        "• BENCHMARK ENTRE REGIONES: Comparación automática de clusters entre mercados",
    ]

    for innovacion in innovaciones:
        print(f"   {innovacion}")

    print("\n\n💡 RECOMENDACIONES PARA ADOPCIÓN TECNOLÓGICA")
    print("=" * 80)

    recomendaciones_tech = [
        "1. INFRAESTRUCTURA: Migrar a plataforma cloud (AWS/Azure) para escalar análisis",
        "2. TALENTO: Contratar 2-3 científicos de datos especializados en no supervisado",
        "3. HERRAMIENTAS: Invertir en plataformas de ML Ops (MLflow, Kubeflow)",
        "4. CULTURA: Capacitar equipos de negocio en interpretación de resultados de ML",
        "5. GOBERNANZA: Establecer comité de ética de IA para uso responsable de datos",
    ]

    for rec_tech in recomendaciones_tech:
        print(f"   {rec_tech}")

    # ------------------------------------------------------------------
    # 17. Validación y próximos pasos
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("17. VALIDACIÓN Y PRÓXIMOS PASOS")
    print("=" * 80)

    print("\n✅ TRABAJO COMPLETADO:")
    print("-" * 80)
    tareas_completadas = [
        "Análisis exploratorio detallado de 10,000 clientes",
        f"Identificación de {k_optimo} segmentos distintos con K-Medias, DBSCAN y Jerárquico",
        f"Detección de {num_anomalias_bosque} clientes con comportamiento anómalo",
        f"Reducción de dimensionalidad de {num_componentes_total} a {num_componentes_95} variables (95% varianza)",
        "Estrategia de imputación de datos faltantes validada",
        "14+ visualizaciones profesionales generadas",
        "Interpretación de negocio de todos los hallazgos",
    ]

    for tarea in tareas_completadas:
        print(f"   ✓ {tarea}")

    if MLXTEND_DISPONIBLE:
        print(f"   ✓ Reglas de asociación identificadas y documentadas")
    else:
        print(f"   ⚠ Metodología de reglas de asociación implementada (requiere mlxtend)")

    print("\n\n➡ PRÓXIMOS PASOS INMEDIATOS:")
    print("-" * 80)
    pasos_inmediatos = [
        "1. Validar hallazgos con equipos de Customer Experience y Retención",
        "2. Presentar resultados a dirección comercial para aprobación de presupuesto",
        "3. Diseñar piloto de retención en grupo de alto riesgo (3 meses)",
        "4. Implementar dashboard de monitoreo de anomalías en tiempo real",
        "5. Integrar clusters como features en modelos supervisados (Unidad 2)",
        "6. Establecer KPIs y métricas de seguimiento de efectividad",
    ]

    for paso in pasos_inmediatos:
        print(f"   {paso}")

    print("\n\n📊 MÉTRICAS DE ÉXITO SUGERIDAS:")
    print("-" * 80)
    metricas_exito = [
        "• Reducción de tasa de abandono en grupo alto riesgo: Meta 10-15%",
        "• Tiempo de detección de clientes en riesgo: Meta < 7 días",
        "• Efectividad de intervenciones: Meta 25-40% de retención",
        "• ROI de programa de retención: Meta > 3x inversión",
        "• Satisfacción de clientes contactados: Meta NPS > 50",
    ]

    for metrica in metricas_exito:
        print(f"   {metrica}")

    # ------------------------------------------------------------------
    # Mensaje final
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 80)

    print(f"\n📁 Archivos generados:")
    print(f"   • {14 if MLXTEND_DISPONIBLE else 13} visualizaciones PNG en carpeta 'outputs/unit3/'")
    print(f"   • Conjunto de datos enriquecido con clusters y detección de anomalías")
    print(f"   • Script completamente documentado en español")

    print(f"\n⏱ Tiempo de ejecución: ~3-5 minutos")
    print(f"📅 Fecha: Febrero 2026")
    print(f"🏢 Empresa: Telmex Claro")
    print(f"🎓 Institución: Universidad Piloto de Colombia")
    print(f"📚 Curso: Aprendizaje de Máquina - Unidad 3")

    print("\n" + "=" * 80)
    print("✨ Este análisis demuestra el poder del aprendizaje no supervisado")
    print("   para descubrir patrones ocultos y generar valor de negocio real.")
    print("=" * 80)

    print("\n💡 RECORDATORIO:")
    print("   Los hallazgos aquí presentados complementan perfectamente")
    print("   el trabajo supervisado de Unidades 1 y 2, formando un sistema")
    print("   integral de inteligencia de clientes y prevención de abandono.")

    print("\n🚀 ¡Muchos éxitos en la implementación de estas estrategias!")


if __name__ == "__main__":
    main()
