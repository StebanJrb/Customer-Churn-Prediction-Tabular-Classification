"""Unidad 3: reduccion de dimensionalidad -- ACP (PCA) y t-SNE."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registra la proyeccion 3d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from churn.viz import output_dir

COLORES_GRUPOS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']


def aplicar_pca_2d(X_escalado, semilla: int):
    """ACP a 2 componentes, usado solo para visualizar los grupos de agrupamiento."""
    acp_viz = PCA(n_components=2, random_state=semilla)
    X_acp = acp_viz.fit_transform(X_escalado)
    return X_acp, acp_viz


def aplicar_pca_completo(X_escalado, semilla: int):
    """ACP con todas las componentes; calcula varianza explicada/acumulada."""
    print("\n🔍 8.1 Aplicando ACP a todas las dimensiones")
    print(f"    Dimensiones originales: {X_escalado.shape[1]}")

    modelo_acp_completo = PCA(random_state=semilla)
    X_acp_completo = modelo_acp_completo.fit_transform(X_escalado)

    varianza_explicada = modelo_acp_completo.explained_variance_ratio_
    varianza_acumulada = np.cumsum(varianza_explicada)

    num_componentes_total = len(varianza_explicada)
    num_componentes_95 = np.argmax(varianza_acumulada >= 0.95) + 1

    print(f"\n📊 Resultados del ACP:")
    print("=" * 60)
    print(f"  • Número total de componentes: {num_componentes_total}")
    print(f"  • Varianza explicada por primeros 5 CP: {varianza_acumulada[4]*100:.2f}%")
    print(f"  • Componentes para 95% de varianza: {num_componentes_95}")
    print(f"  • Reducción dimensional: {num_componentes_total} → {num_componentes_95} variables")
    print(f"  • Compresión lograda: {(1 - num_componentes_95/num_componentes_total)*100:.1f}%")
    print("=" * 60)

    return modelo_acp_completo, X_acp_completo, varianza_explicada, varianza_acumulada, num_componentes_total, num_componentes_95


def graficar_pca_varianza(varianza_explicada, varianza_acumulada, num_componentes_95: int):
    """Barra de varianza por componente + curva acumulada (guarda 09_acp_varianza_explicada.png)."""
    print("\n📊 Generando gráficos de varianza explicada...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(range(1, len(varianza_explicada)+1), varianza_explicada,
           alpha=0.8, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Componente Principal', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Proporción de Varianza Explicada', fontsize=12, fontweight='bold')
    ax1.set_title('Varianza Explicada por Cada Componente Principal',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim([0, len(varianza_explicada)+1])

    ax2.plot(range(1, len(varianza_acumulada)+1), varianza_acumulada,
            marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
    ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% de varianza')
    ax2.axvline(x=num_componentes_95, color='green', linestyle='--', linewidth=2,
               label=f'{num_componentes_95} componentes')
    ax2.set_xlabel('Número de Componentes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Varianza Acumulada', fontsize=12, fontweight='bold')
    ax2.set_title('Varianza Acumulada por Componentes Principales',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "09_acp_varianza_explicada.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 09_acp_varianza_explicada.png")


def graficar_pca_3d(X_acp_completo, varianza_explicada, grupos_kmeans):
    """Scatter 3D de los primeros 3 componentes, coloreado por grupo K-Medias (guarda 10_acp_3d.png)."""
    print("\n📊 8.2 Visualización 3D con primeros 3 componentes principales")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    colores_3d = [COLORES_GRUPOS[i] for i in grupos_kmeans]
    ax.scatter(X_acp_completo[:, 0], X_acp_completo[:, 1], X_acp_completo[:, 2],
               c=colores_3d, alpha=0.6, edgecolors='k', linewidth=0.3, s=30)

    ax.set_xlabel(f'CP1 ({varianza_explicada[0]*100:.1f}% var.)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'CP2 ({varianza_explicada[1]*100:.1f}% var.)', fontsize=11, fontweight='bold')
    ax.set_zlabel(f'CP3 ({varianza_explicada[2]*100:.1f}% var.)', fontsize=11, fontweight='bold')
    ax.set_title('Datos en Espacio de 3 Componentes Principales\n(Coloreado por Grupo K-Medias)',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "10_acp_3d.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 10_acp_3d.png")


def analizar_cargas_pca(modelo_acp_completo, nombres_variables, varianza_explicada):
    """Imprime las variables con mayor carga (loading) en los primeros 3 componentes."""
    print("\n📋 8.3 VARIABLES MÁS IMPORTANTES EN PRIMEROS 3 COMPONENTES")
    print("=" * 80)

    cargas = modelo_acp_completo.components_[:3, :]

    for idx_cp in range(3):
        print(f"\n🔹 Componente Principal {idx_cp+1} (Varianza: {varianza_explicada[idx_cp]*100:.2f}%):")
        print("-" * 70)
        cargas_cp = pd.Series(cargas[idx_cp], index=nombres_variables)
        variables_importantes = cargas_cp.abs().sort_values(ascending=False).head(5)

        for variable, valor_abs in variables_importantes.items():
            valor_real = cargas_cp[variable]
            direccion = "+" if valor_real > 0 else "-"
            print(f"  {direccion} {variable}: {valor_real:.3f}")


def aplicar_tsne(X_escalado, semilla: int):
    """t-SNE 2D para visualizacion de estructura no lineal."""
    print("\n🔍 9.1 Aplicando t-SNE para visualización en 2D")
    print("    ⏳ Esto puede tomar 1-2 minutos (algoritmo iterativo)...")
    print("    Parámetros: perplexity=30, max_iter=1000")

    modelo_tsne = TSNE(n_components=2, random_state=semilla, perplexity=30, max_iter=1000, verbose=0)
    X_tsne = modelo_tsne.fit_transform(X_escalado)

    print("✓ t-SNE completado exitosamente")
    return X_tsne


def graficar_tsne(X_tsne, grupos_kmeans, k_optimo: int, df_clientes: pd.DataFrame):
    """Dos vistas t-SNE: por grupo K-Medias y por abandono (guarda 11_tsne_visualizacion.png)."""
    print("\n📊 Generando visualizaciones t-SNE (por grupo y por abandono)...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for grupo_id in range(k_optimo):
        mascara = grupos_kmeans == grupo_id
        ax1.scatter(X_tsne[mascara, 0], X_tsne[mascara, 1],
                   c=COLORES_GRUPOS[grupo_id], label=f'Grupo {grupo_id}',
                   alpha=0.6, edgecolors='k', linewidth=0.3, s=40)

    ax1.set_xlabel('t-SNE Dimensión 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('t-SNE Dimensión 2', fontsize=12, fontweight='bold')
    ax1.set_title('t-SNE Coloreado por Grupo K-Medias', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    colores_abandono = ['#2ECC71', '#E74C3C']
    for valor_abandono in [0, 1]:
        mascara = df_clientes['abandono_30'] == valor_abandono
        etiqueta = 'Sin Abandono' if valor_abandono == 0 else 'Con Abandono'
        ax2.scatter(X_tsne[mascara, 0], X_tsne[mascara, 1],
                   c=colores_abandono[valor_abandono], label=etiqueta,
                   alpha=0.6, edgecolors='k', linewidth=0.3, s=40)

    ax2.set_xlabel('t-SNE Dimensión 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('t-SNE Dimensión 2', fontsize=12, fontweight='bold')
    ax2.set_title('t-SNE Coloreado por Abandono de Clientes', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Visualización t-SNE: Estructura No Lineal de los Datos',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "11_tsne_visualizacion.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 11_tsne_visualizacion.png")

    print("\n✅ INTERPRETACIÓN t-SNE:")
    print("  • t-SNE revela la estructura no lineal de los datos")
    print("  • Los grupos K-Medias se visualizan claramente separados")
    print("  • La separación entre clientes con/sin abandono es visible")
    print("  • Útil para comunicar hallazgos a audiencias no técnicas")
