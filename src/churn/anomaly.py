"""Unidad 3: deteccion de anomalias -- Bosque de Aislamiento y Factor de Outlier Local."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from churn.viz import output_dir


def entrenar_isolation_forest(X_escalado, df_clientes: pd.DataFrame, semilla: int):
    """Entrena IsolationForest (contamination=5%) y anota anomalia_bosque en df_clientes."""
    print("\n🎯 10.1 Entrenando Bosque de Aislamiento")
    print("    Parámetros: contamination=5%, n_estimators=100")

    modelo_bosque_aislamiento = IsolationForest(
        contamination=0.05,
        random_state=semilla,
        n_estimators=100
    )
    anomalias_bosque = modelo_bosque_aislamiento.fit_predict(X_escalado)

    df_clientes = df_clientes.copy()
    df_clientes['anomalia_bosque'] = anomalias_bosque

    num_anomalias_bosque = (anomalias_bosque == -1).sum()
    print(f"\n📊 Resultados Bosque de Aislamiento:")
    print("=" * 60)
    print(f"  • Anomalías detectadas: {num_anomalias_bosque} ({num_anomalias_bosque/len(df_clientes)*100:.2f}%)")
    print(f"  • Clientes normales: {(anomalias_bosque == 1).sum()} ({(anomalias_bosque == 1).sum()/len(df_clientes)*100:.2f}%)")
    print("=" * 60)

    return df_clientes, anomalias_bosque, num_anomalias_bosque


def analizar_anomalias_bosque(df_clientes: pd.DataFrame):
    """Compara caracteristicas promedio de anomalias vs clientes normales (Bosque de Aislamiento)."""
    print("\n📈 10.2 CARACTERÍSTICAS DE ANOMALÍAS VS CLIENTES NORMALES")
    print("=" * 80)

    df_anomalias_bosque = df_clientes[df_clientes['anomalia_bosque'] == -1]
    df_normales_bosque = df_clientes[df_clientes['anomalia_bosque'] == 1]

    variables_analisis = ['antiguedad', 'cargo_mensual', 'num_quejas', 'num_tickets',
                         'dias_mora', 'pagos_tardios', 'cambio_consumo', 'abandono_30']

    print(f"\n{'Variable':<20} {'Anomalías':<15} {'Normales':<15} {'Diferencia %':<15} {'Significancia'}")
    print("-" * 95)

    for variable in variables_analisis:
        promedio_anomalia = df_anomalias_bosque[variable].mean()
        promedio_normal = df_normales_bosque[variable].mean()
        diferencia = ((promedio_anomalia - promedio_normal) / promedio_normal * 100) if promedio_normal != 0 else 0

        if abs(diferencia) > 20:
            significancia = "⚠ MUY ALTA"
        elif abs(diferencia) > 10:
            significancia = "📊 ALTA"
        elif abs(diferencia) > 5:
            significancia = "📈 MEDIA"
        else:
            significancia = "→ BAJA"

        print(f"{variable:<20} {promedio_anomalia:<15.2f} {promedio_normal:<15.2f} {diferencia:+<15.1f} {significancia}")

    tasa_abandono_anomalias = df_anomalias_bosque['abandono_30'].mean() * 100
    tasa_abandono_normales = df_normales_bosque['abandono_30'].mean() * 100
    ratio_abandono = tasa_abandono_anomalias / tasa_abandono_normales if tasa_abandono_normales > 0 else 0

    print("\n" + "=" * 80)
    print("🎯 INTERPRETACIÓN CLAVE:")
    print("=" * 80)
    print(f"  🔴 Tasa de abandono en anomalías: {tasa_abandono_anomalias:.1f}%")
    print(f"  🟢 Tasa de abandono en normales: {tasa_abandono_normales:.1f}%")
    print(f"  ⚡ Las anomalías tienen {ratio_abandono:.1f}x más riesgo de abandono")
    print("=" * 80)

    return df_anomalias_bosque, df_normales_bosque, tasa_abandono_anomalias, tasa_abandono_normales, ratio_abandono


def graficar_anomalias_bosque(X_tsne, anomalias_bosque, num_anomalias_bosque):
    """Scatter t-SNE de anomalias vs normales del Bosque de Aislamiento (guarda 12_bosque_aislamiento_anomalias.png)."""
    print("\n📊 Generando visualización de anomalías...")
    plt.figure(figsize=(12, 8))

    mascara_normales = anomalias_bosque == 1
    plt.scatter(X_tsne[mascara_normales, 0], X_tsne[mascara_normales, 1],
               c='#3498DB', label='Clientes Normales', alpha=0.5, s=30)

    mascara_anomalias = anomalias_bosque == -1
    plt.scatter(X_tsne[mascara_anomalias, 0], X_tsne[mascara_anomalias, 1],
               c='#E74C3C', label='Anomalías Detectadas', alpha=0.8, s=80,
               edgecolors='black', linewidth=1, marker='X')

    plt.xlabel('t-SNE Dimensión 1', fontsize=12, fontweight='bold')
    plt.ylabel('t-SNE Dimensión 2', fontsize=12, fontweight='bold')
    plt.title(f'Detección de Anomalías: Bosque de Aislamiento\n({num_anomalias_bosque} anomalías identificadas)',
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "12_bosque_aislamiento_anomalias.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 12_bosque_aislamiento_anomalias.png")


def entrenar_lof(X_escalado, df_clientes: pd.DataFrame):
    """Entrena LocalOutlierFactor (contamination=5%, n_neighbors=20) y anota anomalia_lof."""
    print("\n🎯 11.1 Entrenando LOF")
    print("    Parámetros: contamination=5%, n_neighbors=20")

    modelo_lof = LocalOutlierFactor(contamination=0.05, n_neighbors=20)
    anomalias_lof = modelo_lof.fit_predict(X_escalado)

    df_clientes = df_clientes.copy()
    df_clientes['anomalia_lof'] = anomalias_lof

    num_anomalias_lof = (anomalias_lof == -1).sum()
    print(f"\n📊 Resultados LOF:")
    print("=" * 60)
    print(f"  • Anomalías detectadas: {num_anomalias_lof} ({num_anomalias_lof/len(df_clientes)*100:.2f}%)")
    print(f"  • Clientes normales: {(anomalias_lof == 1).sum()} ({(anomalias_lof == 1).sum()/len(df_clientes)*100:.2f}%)")
    print("=" * 60)

    return df_clientes, anomalias_lof, num_anomalias_lof


def comparar_bosque_lof(df_clientes: pd.DataFrame, num_anomalias_bosque):
    """Matriz de concordancia entre Bosque de Aislamiento y LOF."""
    print("\n🔍 11.2 COMPARACIÓN: BOSQUE DE AISLAMIENTO VS LOF")
    print("=" * 80)

    ambos_anomalias = ((df_clientes['anomalia_bosque'] == -1) & (df_clientes['anomalia_lof'] == -1)).sum()
    solo_bosque = ((df_clientes['anomalia_bosque'] == -1) & (df_clientes['anomalia_lof'] == 1)).sum()
    solo_lof = ((df_clientes['anomalia_bosque'] == 1) & (df_clientes['anomalia_lof'] == -1)).sum()
    ambos_normales = ((df_clientes['anomalia_bosque'] == 1) & (df_clientes['anomalia_lof'] == 1)).sum()

    print(f"\n📊 Matriz de Concordancia:")
    print("=" * 60)
    print(f"  • Detectadas por AMBOS métodos: {ambos_anomalias} ({ambos_anomalias/len(df_clientes)*100:.1f}%)")
    print(f"  • Solo Bosque de Aislamiento: {solo_bosque} ({solo_bosque/len(df_clientes)*100:.1f}%)")
    print(f"  • Solo LOF: {solo_lof} ({solo_lof/len(df_clientes)*100:.1f}%)")
    print(f"  • Normales en ambos: {ambos_normales} ({ambos_normales/len(df_clientes)*100:.1f}%)")
    print("=" * 60)

    concordancia = (ambos_anomalias + ambos_normales) / len(df_clientes) * 100
    print(f"\n✅ Concordancia total entre métodos: {concordancia:.1f}%")

    return ambos_anomalias, solo_bosque, solo_lof, ambos_normales, concordancia


def graficar_comparacion_anomalias(X_tsne, anomalias_bosque, anomalias_lof, num_anomalias_bosque, num_anomalias_lof):
    """Scatter t-SNE lado a lado: Bosque de Aislamiento vs LOF (guarda 13_comparacion_anomalias.png)."""
    print("\n📊 Generando visualización comparativa...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for valor in [1, -1]:
        mascara = anomalias_bosque == valor
        color = '#3498DB' if valor == 1 else '#E74C3C'
        etiqueta = 'Normal' if valor == 1 else 'Anomalía'
        marker = 'o' if valor == 1 else 'X'
        tamaño = 30 if valor == 1 else 80

        ax1.scatter(X_tsne[mascara, 0], X_tsne[mascara, 1],
                   c=color, label=etiqueta, alpha=0.6 if valor == 1 else 0.8,
                   s=tamaño, edgecolors='black' if valor == -1 else 'none',
                   linewidth=1, marker=marker)

    ax1.set_xlabel('t-SNE Dimensión 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('t-SNE Dimensión 2', fontsize=12, fontweight='bold')
    ax1.set_title(f'Bosque de Aislamiento\n{num_anomalias_bosque} anomalías',
                 fontsize=13, fontweight='bold', pad=15)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    for valor in [1, -1]:
        mascara = anomalias_lof == valor
        color = '#3498DB' if valor == 1 else '#E74C3C'
        etiqueta = 'Normal' if valor == 1 else 'Anomalía'
        marker = 'o' if valor == 1 else 'X'
        tamaño = 30 if valor == 1 else 80

        ax2.scatter(X_tsne[mascara, 0], X_tsne[mascara, 1],
                   c=color, label=etiqueta, alpha=0.6 if valor == 1 else 0.8,
                   s=tamaño, edgecolors='black' if valor == -1 else 'none',
                   linewidth=1, marker=marker)

    ax2.set_xlabel('t-SNE Dimensión 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('t-SNE Dimensión 2', fontsize=12, fontweight='bold')
    ax2.set_title(f'Factor de Outlier Local (LOF)\n{num_anomalias_lof} anomalías',
                 fontsize=13, fontweight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Comparación de Métodos de Detección de Anomalías',
                fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir("unit3") / "13_comparacion_anomalias.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 13_comparacion_anomalias.png")
