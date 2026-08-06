"""Unidad 3: agrupamiento avanzado -- K-Medias, DBSCAN y jerarquico (Ward).

Distinto de churn.clustering.perform_clustering (K-Means basico de Unidad 1),
este modulo evalua varios algoritmos de agrupamiento (K-Medias, DBSCAN,
jerarquico aglomerativo) sobre los datos preparados en Unidad 3 (imputados +
codificados One-Hot + escalados).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

from churn.viz import output_dir

COLORES_GRUPOS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']


def preparar_datos_agrupamiento(df_clientes: pd.DataFrame, df_limpio: pd.DataFrame):
    """One-Hot Encoding de categoricas + escalamiento estandar para agrupamiento.

    Returns
    -------
    tuple
        (df_agrupamiento, df_escalado, X_escalado, escalador, dummies_contrato, dummies_pago)
    """
    df_para_agrupar = df_clientes.copy()

    print("\n🔧 Aplicando codificación One-Hot a variables categóricas...")
    dummies_contrato = pd.get_dummies(df_para_agrupar['tipo_contrato'], prefix='contrato')
    dummies_pago = pd.get_dummies(df_para_agrupar['metodo_pago'], prefix='pago')

    print(f"  ✓ tipo_contrato → {len(dummies_contrato.columns)} variables binarias")
    print(f"  ✓ metodo_pago → {len(dummies_pago.columns)} variables binarias")

    df_agrupamiento_completo = pd.concat([df_limpio, dummies_contrato, dummies_pago], axis=1)
    df_agrupamiento = df_agrupamiento_completo.drop(['abandono_30'], axis=1, errors='ignore')

    print(f"\n✓ Conjunto de datos para agrupamiento: {df_agrupamiento.shape}")
    print(f"✓ Variables totales: {len(df_agrupamiento.columns)}")
    print(f"  - Numéricas originales: 11")
    print(f"  - Binarias (One-Hot): {len(dummies_contrato.columns) + len(dummies_pago.columns)}")

    print("\n🔧 Aplicando escalamiento estándar (StandardScaler)...")
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(df_agrupamiento)
    df_escalado = pd.DataFrame(X_escalado, columns=df_agrupamiento.columns)

    print("✓ Datos normalizados exitosamente (media=0, desviación estándar=1)")
    print("\n📊 Estadísticas del conjunto escalado:")
    print(df_escalado.describe().loc[['mean', 'std']])

    return df_agrupamiento, df_escalado, X_escalado, escalador, dummies_contrato, dummies_pago


def seleccionar_k_kmeans(X_escalado, semilla: int, rango_k=range(2, 11)):
    """Metodo del codo + coeficiente de silueta para elegir k (guarda 03_kmeans_codo_silueta.png)."""
    print("\n🔍 5.1 Determinando número óptimo de grupos (k)")
    print("     Método del Codo + Coeficiente de Silueta")

    inercias = []
    coeficientes_silueta = []

    print("\nCalculando métricas para k de 2 a 10...")
    for k in rango_k:
        modelo_kmeans = KMeans(n_clusters=k, random_state=semilla, n_init=10)
        modelo_kmeans.fit(X_escalado)
        inercias.append(modelo_kmeans.inertia_)
        silueta = silhouette_score(X_escalado, modelo_kmeans.labels_)
        coeficientes_silueta.append(silueta)
        print(f"  k={k}: Inercia={modelo_kmeans.inertia_:.0f}, Silueta={silueta:.3f}")

    print("\n📊 Generando gráficos de selección de k...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    rango_k_lista = list(rango_k)

    ax1.plot(rango_k_lista, inercias, marker='o', linewidth=2.5, markersize=10, color='#2E86AB')
    ax1.set_xlabel('Número de Grupos (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inercia (Suma de Cuadrados Intra-Grupo)', fontsize=12, fontweight='bold')
    ax1.set_title('Método del Codo para Selección de k', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axvline(x=4, color='red', linestyle='--', linewidth=2, label='k óptimo = 4')
    ax1.legend(fontsize=11)

    ax2.plot(rango_k_lista, coeficientes_silueta, marker='s', color='#06A77D', linewidth=2.5, markersize=10)
    ax2.set_xlabel('Número de Grupos (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Coeficiente de Silueta', fontsize=12, fontweight='bold')
    ax2.set_title('Coeficiente de Silueta por k', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axvline(x=4, color='red', linestyle='--', linewidth=2, label='k óptimo = 4')
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "03_kmeans_codo_silueta.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 03_kmeans_codo_silueta.png")

    return inercias, coeficientes_silueta


def entrenar_kmeans(X_escalado, df_clientes: pd.DataFrame, k_optimo: int, semilla: int):
    """Entrena K-Medias final, calcula metricas de calidad y perfila/nombra los grupos."""
    print(f"\n🎯 5.2 Entrenando K-Medias con k={k_optimo}")
    modelo_kmeans_final = KMeans(n_clusters=k_optimo, random_state=semilla, n_init=10)
    grupos_kmeans = modelo_kmeans_final.fit_predict(X_escalado)

    silueta_promedio = silhouette_score(X_escalado, grupos_kmeans)
    davies_bouldin = davies_bouldin_score(X_escalado, grupos_kmeans)
    calinski_harabasz = calinski_harabasz_score(X_escalado, grupos_kmeans)

    print(f"\n📊 Métricas de Calidad del Agrupamiento:")
    print("=" * 60)
    print(f"  • Coeficiente de Silueta: {silueta_promedio:.3f}")
    print(f"    (Rango: -1 a 1, mejor cercano a 1)")
    print(f"  • Índice Davies-Bouldin: {davies_bouldin:.3f}")
    print(f"    (Menor es mejor, indica compacidad y separación)")
    print(f"  • Puntaje Calinski-Harabasz: {calinski_harabasz:.1f}")
    print(f"    (Mayor es mejor, indica densidad y separación)")
    print("=" * 60)

    df_clientes = df_clientes.copy()
    df_clientes['grupo_kmeans'] = grupos_kmeans

    print("\n📊 Distribución de clientes por grupo:")
    distribucion_grupos = df_clientes['grupo_kmeans'].value_counts().sort_index()
    for grupo_id, cantidad in distribucion_grupos.items():
        porcentaje = cantidad / len(df_clientes) * 100
        print(f"  Grupo {grupo_id}: {cantidad:,} clientes ({porcentaje:.1f}%)")

    print("\n📈 5.3 PERFIL DETALLADO DE CADA GRUPO")
    print("=" * 80)

    perfil_grupos = df_clientes.groupby('grupo_kmeans')[
        ['antiguedad', 'cargo_mensual', 'num_quejas', 'num_tickets', 'dias_mora', 'pagos_tardios', 'abandono_30']
    ].mean()

    print("\nPromedios por grupo:")
    print(perfil_grupos.round(2))

    print("\n" + "=" * 80)
    print("🎯 INTERPRETACIÓN DE NEGOCIO DE LOS GRUPOS")
    print("=" * 80)

    nombres_grupos = {}

    for grupo_id in range(k_optimo):
        datos_grupo = df_clientes[df_clientes['grupo_kmeans'] == grupo_id]
        tasa_abandono = datos_grupo['abandono_30'].mean() * 100
        promedio_antiguedad = datos_grupo['antiguedad'].mean()
        promedio_cargo = datos_grupo['cargo_mensual'].mean()
        promedio_quejas = datos_grupo['num_quejas'].mean()
        promedio_mora = datos_grupo['dias_mora'].mean()
        contrato_principal = datos_grupo['tipo_contrato'].mode()[0]

        print(f"\n🔸 GRUPO {grupo_id}:")
        print(f"{'='*70}")
        print(f"  📊 Tamaño: {len(datos_grupo):,} clientes ({len(datos_grupo)/len(df_clientes)*100:.1f}%)")
        print(f"  ⚠ Tasa de abandono: {tasa_abandono:.1f}%")
        print(f"  📅 Antigüedad promedio: {promedio_antiguedad:.1f} meses")
        print(f"  💰 Cargo mensual promedio: ${promedio_cargo:,.0f} COP")
        print(f"  📝 Quejas promedio: {promedio_quejas:.2f}")
        print(f"  ⏰ Días de mora promedio: {promedio_mora:.1f}")
        print(f"  📋 Contrato predominante: {contrato_principal}")

        if tasa_abandono > 70:
            nivel_riesgo = "🔴 RIESGO CRÍTICO"
            nombre = "Clientes en Riesgo Extremo"
            print(f"  >>> {nivel_riesgo}")
            print(f"  >>> ACCIÓN: Intervención INMEDIATA - Equipo dedicado 24/7")
            print(f"  >>> ESTRATEGIA: Ofertas agresivas de retención + Resolución express de problemas")
        elif tasa_abandono > 50:
            nivel_riesgo = "🟠 RIESGO ALTO"
            nombre = "Clientes Insatisfechos"
            print(f"  >>> {nivel_riesgo}")
            print(f"  >>> ACCIÓN: Intervención urgente dentro de 72 horas")
            print(f"  >>> ESTRATEGIA: Llamadas proactivas + Incentivos personalizados")
        elif tasa_abandono > 30:
            nivel_riesgo = "🟡 RIESGO MEDIO"
            nombre = "Clientes en Observación"
            print(f"  >>> {nivel_riesgo}")
            print(f"  >>> ACCIÓN: Monitoreo cercano semanal")
            print(f"  >>> ESTRATEGIA: Programas de fidelización + Encuestas de satisfacción")
        else:
            nivel_riesgo = "🟢 BAJO RIESGO"
            nombre = "Clientes Estables y Satisfechos"
            print(f"  >>> {nivel_riesgo}")
            print(f"  >>> ACCIÓN: Mantener calidad de servicio")
            print(f"  >>> ESTRATEGIA: Programas de referidos + Oportunidades de upselling")

        nombres_grupos[grupo_id] = nombre

    return df_clientes, modelo_kmeans_final, grupos_kmeans, nombres_grupos, silueta_promedio, davies_bouldin, calinski_harabasz


def graficar_kmeans_pca(X_acp, modelo_acp_2d, modelo_kmeans_final, grupos_kmeans, k_optimo: int, nombres_grupos: dict):
    """Visualiza los grupos de K-Medias en el espacio ACP 2D (guarda 04_kmeans_grupos_2d.png)."""
    print("\n📉 5.4 Visualización de grupos en espacio 2D usando ACP")
    plt.figure(figsize=(12, 8))
    for grupo_id in range(k_optimo):
        mascara = grupos_kmeans == grupo_id
        plt.scatter(X_acp[mascara, 0], X_acp[mascara, 1],
                   c=COLORES_GRUPOS[grupo_id], label=f'Grupo {grupo_id}: {nombres_grupos[grupo_id]}',
                   alpha=0.6, edgecolors='k', linewidth=0.5, s=50)

    centroides_acp = modelo_acp_2d.transform(modelo_kmeans_final.cluster_centers_)
    plt.scatter(centroides_acp[:, 0], centroides_acp[:, 1], c='red', marker='X', s=400,
               edgecolors='black', linewidth=3, label='Centroides', zorder=100)

    plt.xlabel(f'Componente Principal 1 ({modelo_acp_2d.explained_variance_ratio_[0]*100:.1f}% varianza)',
              fontsize=12, fontweight='bold')
    plt.ylabel(f'Componente Principal 2 ({modelo_acp_2d.explained_variance_ratio_[1]*100:.1f}% varianza)',
              fontsize=12, fontweight='bold')
    plt.title('Grupos K-Medias Visualizados con Análisis de Componentes Principales',
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "04_kmeans_grupos_2d.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 04_kmeans_grupos_2d.png")


def seleccionar_eps_dbscan(X_escalado):
    """Grafico de k-distancia para elegir eps de DBSCAN (guarda 05_dbscan_seleccion_eps.png)."""
    print("\n🔍 6.1 Determinando parámetro eps óptimo")
    print("    Usando gráfico de k-distancia...")

    vecinos = NearestNeighbors(n_neighbors=5)
    vecinos_ajustados = vecinos.fit(X_escalado)
    distancias, indices = vecinos_ajustados.kneighbors(X_escalado)

    distancias_ordenadas = np.sort(distancias[:, -1], axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(distancias_ordenadas, linewidth=2, color='#2E86AB')
    plt.xlabel('Puntos Ordenados', fontsize=12, fontweight='bold')
    plt.ylabel('Distancia al 5to Vecino Más Cercano', fontsize=12, fontweight='bold')
    plt.title('Gráfico K-Distancia para Determinar eps Óptimo en DBSCAN',
             fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=3.5, color='red', linestyle='--', linewidth=2, label='eps sugerido ≈ 3.5')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "05_dbscan_seleccion_eps.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 05_dbscan_seleccion_eps.png")

    return distancias_ordenadas


def entrenar_dbscan(X_escalado, df_clientes: pd.DataFrame, eps_optimo: float, muestras_minimas: int):
    print(f"\n🎯 6.2 Entrenando DBSCAN")
    print(f"    Parámetros: eps={eps_optimo}, min_samples={muestras_minimas}")

    modelo_dbscan = DBSCAN(eps=eps_optimo, min_samples=muestras_minimas)
    grupos_dbscan = modelo_dbscan.fit_predict(X_escalado)

    df_clientes = df_clientes.copy()
    df_clientes['grupo_dbscan'] = grupos_dbscan

    num_grupos_dbscan = len(set(grupos_dbscan)) - (1 if -1 in grupos_dbscan else 0)
    num_ruido = list(grupos_dbscan).count(-1)

    print(f"\n📊 Resultados de DBSCAN:")
    print("=" * 60)
    print(f"  • Grupos encontrados: {num_grupos_dbscan}")
    print(f"  • Puntos de ruido (outliers): {num_ruido} ({num_ruido/len(df_clientes)*100:.1f}%)")
    print("=" * 60)

    print("\n📊 Distribución de clientes por grupo:")
    distribucion_dbscan = df_clientes['grupo_dbscan'].value_counts().sort_index()
    for grupo_id, cantidad in distribucion_dbscan.items():
        porcentaje = cantidad / len(df_clientes) * 100
        etiqueta = "Ruido/Outliers" if grupo_id == -1 else f"Grupo {grupo_id}"
        print(f"  {etiqueta}: {cantidad:,} clientes ({porcentaje:.1f}%)")

    return df_clientes, grupos_dbscan, num_grupos_dbscan, num_ruido


def graficar_dbscan(X_acp, grupos_dbscan, num_grupos_dbscan: int):
    """Visualiza los grupos de DBSCAN en el espacio ACP 2D (guarda 06_dbscan_grupos.png)."""
    print("\n📉 Generando visualización de DBSCAN...")
    plt.figure(figsize=(12, 8))
    colores_dbscan = plt.cm.Spectral(np.linspace(0, 1, num_grupos_dbscan + 1))

    for grupo_id in set(grupos_dbscan):
        mascara = grupos_dbscan == grupo_id
        if grupo_id == -1:
            plt.scatter(X_acp[mascara, 0], X_acp[mascara, 1],
                       c='black', label='Ruido/Outliers', alpha=0.3, s=20, marker='x')
        else:
            plt.scatter(X_acp[mascara, 0], X_acp[mascara, 1],
                       c=[colores_dbscan[grupo_id]], label=f'Grupo {grupo_id}',
                       alpha=0.7, edgecolors='k', linewidth=0.5, s=50)

    plt.xlabel(f'Componente Principal 1', fontsize=12, fontweight='bold')
    plt.ylabel(f'Componente Principal 2', fontsize=12, fontweight='bold')
    plt.title('Agrupamiento DBSCAN - Basado en Densidad\n(puntos negros = outliers)',
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "06_dbscan_grupos.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 06_dbscan_grupos.png")


def analizar_outliers_dbscan(df_clientes: pd.DataFrame, num_ruido: int):
    print("\n🔍 6.3 ANÁLISIS DETALLADO DE OUTLIERS DETECTADOS")
    print("=" * 80)

    if num_ruido > 0:
        df_outliers = df_clientes[df_clientes['grupo_dbscan'] == -1]
        df_normales = df_clientes[df_clientes['grupo_dbscan'] != -1]

        print(f"\nComparación: Outliers vs Clientes Normales")
        print("=" * 80)

        variables_comparacion = ['antiguedad', 'cargo_mensual', 'num_quejas', 'num_tickets',
                                'dias_mora', 'pagos_tardios', 'abandono_30']

        print(f"\n{'Variable':<20} {'Outliers':<15} {'Normales':<15} {'Diferencia %'}")
        print("-" * 80)
        for variable in variables_comparacion:
            promedio_outlier = df_outliers[variable].mean()
            promedio_normal = df_normales[variable].mean()
            diferencia = ((promedio_outlier - promedio_normal) / promedio_normal * 100) if promedio_normal != 0 else 0
            print(f"{variable:<20} {promedio_outlier:<15.2f} {promedio_normal:<15.2f} {diferencia:+.1f}%")

        print("\n✅ INTERPRETACIÓN:")
        tasa_abandono_outliers = df_outliers['abandono_30'].mean() * 100
        tasa_abandono_normales = df_normales['abandono_30'].mean() * 100
        print(f"  • Tasa de abandono en outliers: {tasa_abandono_outliers:.1f}%")
        print(f"  • Tasa de abandono en normales: {tasa_abandono_normales:.1f}%")
        if tasa_abandono_outliers > tasa_abandono_normales:
            print(f"  ⚠ Los outliers tienen {tasa_abandono_outliers/tasa_abandono_normales:.1f}x más riesgo de abandono")
    else:
        print("  No se detectaron outliers en este agrupamiento")


def dendrograma_jerarquico(X_escalado, semilla: int, tamaño_muestra: int = 500):
    """Dendrograma con enlace Ward sobre una muestra (guarda 07_dendrograma_jerarquico.png)."""
    print("\n🌳 7.1 Generando dendrograma (muestra de 500 clientes)")
    print("    Nota: Usar muestra para que el dendrograma sea legible")

    np.random.seed(semilla)
    indices_muestra = np.random.choice(len(X_escalado), tamaño_muestra, replace=False)
    X_muestra = X_escalado[indices_muestra]

    print("\n🔧 Calculando matriz de enlace (método Ward)...")
    matriz_enlace = linkage(X_muestra, method='ward')

    plt.figure(figsize=(16, 8))
    dendrogram(matriz_enlace, no_labels=True, color_threshold=50)
    plt.xlabel('Índice de Muestra de Cliente', fontsize=12, fontweight='bold')
    plt.ylabel('Distancia de Enlace (Ward)', fontsize=12, fontweight='bold')
    plt.title(f'Dendrograma de Agrupamiento Jerárquico\n(Muestra de {tamaño_muestra} clientes - Método Ward)',
             fontsize=14, fontweight='bold', pad=20)
    plt.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Nivel de corte sugerido')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "07_dendrograma_jerarquico.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 07_dendrograma_jerarquico.png")


def entrenar_jerarquico(X_escalado, df_clientes: pd.DataFrame, k_optimo: int):
    print(f"\n🎯 7.2 Entrenando Agrupamiento Jerárquico Aglomerativo")
    print(f"    Parámetros: n_clusters={k_optimo}, linkage='ward'")

    modelo_jerarquico = AgglomerativeClustering(n_clusters=k_optimo, linkage='ward')
    grupos_jerarquico = modelo_jerarquico.fit_predict(X_escalado)

    df_clientes = df_clientes.copy()
    df_clientes['grupo_jerarquico'] = grupos_jerarquico

    print("\n📊 Distribución de clientes por grupo:")
    distribucion_jerarquico = df_clientes['grupo_jerarquico'].value_counts().sort_index()
    for grupo_id, cantidad in distribucion_jerarquico.items():
        porcentaje = cantidad / len(df_clientes) * 100
        print(f"  Grupo {grupo_id}: {cantidad:,} clientes ({porcentaje:.1f}%)")

    return df_clientes, grupos_jerarquico


def graficar_jerarquico(X_acp, grupos_jerarquico, k_optimo: int):
    """Visualiza los grupos jerarquicos en el espacio ACP 2D (guarda 08_grupos_jerarquicos.png)."""
    print("\n📉 Generando visualización de grupos jerárquicos...")
    plt.figure(figsize=(12, 8))

    for grupo_id in range(k_optimo):
        mascara = grupos_jerarquico == grupo_id
        plt.scatter(X_acp[mascara, 0], X_acp[mascara, 1],
                   c=COLORES_GRUPOS[grupo_id], label=f'Grupo {grupo_id}',
                   alpha=0.6, edgecolors='k', linewidth=0.5, s=50)

    plt.xlabel(f'Componente Principal 1', fontsize=12, fontweight='bold')
    plt.ylabel(f'Componente Principal 2', fontsize=12, fontweight='bold')
    plt.title('Agrupamiento Jerárquico Aglomerativo (Método Ward)',
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "08_grupos_jerarquicos.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 08_grupos_jerarquicos.png")
