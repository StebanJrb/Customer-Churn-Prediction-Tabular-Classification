"""Unidad 3: reglas de asociacion (Apriori) via mlxtend -- opcional.

mlxtend no es una dependencia dura del proyecto: si no esta instalado, esta
seccion se omite y el pipeline completo sigue corriendo con un mensaje
informativo, replicando el patron try/except del script original.
"""

import matplotlib.pyplot as plt
import pandas as pd

from churn.viz import output_dir

try:
    from mlxtend.frequent_patterns import apriori, association_rules as mlxtend_association_rules
    MLXTEND_DISPONIBLE = True
except ImportError:
    MLXTEND_DISPONIBLE = False
    print("⚠ Advertencia: mlxtend no está instalado. Se omitirán reglas de asociación.")
    print("  Para instalar: pip install mlxtend")


def analizar_reglas_asociacion(df_clientes: pd.DataFrame) -> dict:
    """Binariza variables, ejecuta Apriori + reglas de asociacion si mlxtend esta disponible.

    Si mlxtend no esta disponible imprime la metodologia documentada y no
    genera ningun grafico (14_reglas_asociacion.png se omite).

    Returns
    -------
    dict
        reglas_abandono, reglas_abandono_ordenadas (ambos None si no aplica)
    """
    print("\n" + "=" * 80)
    print("12. REGLAS DE ASOCIACIÓN (ALGORITMO APRIORI)")
    print("=" * 80)

    resultado = {"reglas_abandono": None, "reglas_abandono_ordenadas": None}

    if MLXTEND_DISPONIBLE:
        print("\n✓ mlxt end disponible - Procediendo con análisis de asociación")

        print("\n🔧 12.1 Preparando datos transaccionales (binarización)")

        df_asociacion = df_clientes.copy()

        print("  • Creando variables binarias...")
        df_asociacion['cargo_alto'] = (df_asociacion['cargo_mensual'] > df_asociacion['cargo_mensual'].median()).astype(int)
        df_asociacion['antiguedad_larga'] = (df_asociacion['antiguedad'] > 24).astype(int)
        df_asociacion['antiguedad_corta'] = (df_asociacion['antiguedad'] <= 6).astype(int)
        df_asociacion['muchas_quejas'] = (df_asociacion['num_quejas'] > 2).astype(int)
        df_asociacion['muchos_tickets'] = (df_asociacion['num_tickets'] > 5).astype(int)
        df_asociacion['mora_alta'] = (df_asociacion['dias_mora'] > 30).astype(int)
        df_asociacion['pagador_tardio'] = (df_asociacion['pagos_tardios'] > 3).astype(int)
        df_asociacion['consumo_bajo'] = (df_asociacion['cambio_consumo'] < -20).astype(int)
        df_asociacion['contrato_mes_a_mes'] = (df_asociacion['tipo_contrato'] == 'Mes-a-mes').astype(int)
        df_asociacion['va_a_abandonar'] = df_asociacion['abandono_30']

        variables_binarias = ['cargo_alto', 'antiguedad_larga', 'antiguedad_corta', 'muchas_quejas',
                             'muchos_tickets', 'mora_alta', 'pagador_tardio', 'consumo_bajo',
                             'contrato_mes_a_mes', 'va_a_abandonar']

        df_binario = df_asociacion[variables_binarias]

        print(f"  ✓ {len(variables_binarias)} variables binarias creadas")
        print("\n📊 Primeras 5 transacciones:")
        print(df_binario.head())

        print("\n🔍 12.2 Aplicando algoritmo Apriori")
        print("    Parámetros: min_support=0.05 (5%)")

        conjuntos_frecuentes = apriori(df_binario, min_support=0.05, use_colnames=True)
        print(f"\n✓ Conjuntos frecuentes encontrados: {len(conjuntos_frecuentes)}")

        if len(conjuntos_frecuentes) > 0:
            print("\n📊 Top 10 combinaciones más frecuentes:")
            print(conjuntos_frecuentes.nlargest(10, 'support')[['support', 'itemsets']])

            print("\n🔍 12.3 Generando reglas de asociación")
            print("    Parámetros: metric='confidence', min_threshold=0.3")

            reglas = mlxtend_association_rules(conjuntos_frecuentes, metric="confidence", min_threshold=0.3)
            reglas = reglas.sort_values('lift', ascending=False)

            print(f"\n✓ Reglas generadas: {len(reglas)}")

            print("\n🎯 12.4 REGLAS QUE PREDICEN ABANDONO DE CLIENTES")
            print("=" * 80)

            reglas_abandono = reglas[reglas['consequents'].apply(lambda x: 'va_a_abandonar' in x)]
            reglas_abandono_ordenadas = reglas_abandono.sort_values('lift', ascending=False).head(10)

            if len(reglas_abandono_ordenadas) > 0:
                print(f"\n📊 Top 10 reglas predictoras de abandono:")
                print("=" * 80)

                for idx, fila in reglas_abandono_ordenadas.iterrows():
                    antecedentes = ', '.join(list(fila['antecedents']))
                    print(f"\n  ▶ SI  {antecedentes}")
                    print(f"     ENTONCES → Cliente va a abandonar")
                    print(f"     Support: {fila['support']:.3f} | Confidence: {fila['confidence']:.3f} | Lift: {fila['lift']:.2f}")
                    print(f"     Interpretación: {fila['confidence']*100:.0f}% de clientes con estas características abandonan")
                    print(f"                     (vs {df_clientes['abandono_30'].mean()*100:.0f}% promedio general)")

                print("\n📊 Generando visualización de reglas...")
                plt.figure(figsize=(12, 8))

                scatter = plt.scatter(reglas_abandono['support'], reglas_abandono['confidence'],
                                    s=reglas_abandono['lift']*100, c=reglas_abandono['lift'],
                                    cmap='YlOrRd', alpha=0.7, edgecolors='black', linewidth=1.5)

                plt.colorbar(scatter, label='Lift')
                plt.xlabel('Support (Frecuencia)', fontsize=12, fontweight='bold')
                plt.ylabel('Confidence (Confianza)', fontsize=12, fontweight='bold')
                plt.title('Reglas de Asociación para Predicción de Abandono\n(Tamaño de punto = Lift)',
                         fontsize=14, fontweight='bold', pad=20)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir("unit3") / "14_reglas_asociacion.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ Gráfico guardado: 14_reglas_asociacion.png")

                print("\n✅ CONCLUSIÓN REGLAS DE ASOCIACIÓN:")
                print("=" * 80)
                print(f"  • {len(reglas_abandono)} reglas identifican patrones de abandono")
                print(f"  • Confidence máxima: {reglas_abandono['confidence'].max():.1%}")
                print(f"  • Lift máximo: {reglas_abandono['lift'].max():.1f}x")
                print("  • Útil para crear gatillos automáticos de intervención")
                print("=" * 80)

                resultado["reglas_abandono"] = reglas_abandono
                resultado["reglas_abandono_ordenadas"] = reglas_abandono_ordenadas
            else:
                print("  ⚠ No se encontraron reglas que predigan abandono con los parámetros actuales")
        else:
            print("  ⚠ No se encontraron conjuntos frecuentes con el support mínimo especificado")

    else:
        print("\n⚠ mlxtend NO está disponible")
        print("\n📚 Las reglas de asociación están IMPLEMENTADAS en el código")
        print("   pero requieren la librería mlxtend para ejecutarse.")
        print("\n💡 Para instalar:")
        print("   pip install mlxtend")
        print("\n📋 METODOLOGÍA IMPLEMENTADA:")
        print("   1. Binarización de variables (cargo_alto, muchas_quejas, etc.)")
        print("   2. Algoritmo Apriori con support mínimo 5%")
        print("   3. Generación de reglas con confidence mínima 30%")
        print("   4. Filtrado de reglas que predicen abandono")
        print("   5. Visualización de reglas por support, confidence y lift")
        print("\n✅ La técnica está documentada y lista para ejecutar cuando mlxtend esté disponible")

    return resultado
