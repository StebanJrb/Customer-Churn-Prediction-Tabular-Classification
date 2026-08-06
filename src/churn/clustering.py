"""Unidad 1 baseline: K-Means customer segmentation."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from churn.config import CLUSTER_FEATURES, RANDOM_STATE


def perform_clustering(df: pd.DataFrame, n_clusters: int = 4, random_state: int = RANDOM_STATE):
    """Segment customers with K-Means, then profile and name each cluster.

    Clustering is unsupervised -- churn labels aren't used as inputs, only
    to describe the resulting segments afterward. Cluster names are assigned
    by simple heuristics over each segment's average profile (tenure,
    mora_days, complaints), so results stay easy to hand to a business audience.

    Returns
    -------
    tuple
        (df_with_cluster_columns, fitted_kmeans_model, cluster_profiles)
    """
    X_cluster = df[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    print("\nEvaluacion de numero de clusters:")
    print("-" * 40)
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"  K={k}: Silhouette Score = {score:.3f}")

    kmeans_final = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans_final.fit_predict(X_scaled)

    print(f"\nK seleccionado: {n_clusters}")
    print(f"Silhouette Score final: {silhouette_score(X_scaled, df['cluster']):.3f}")

    print("\n" + "=" * 60)
    print("PERFILES DE CLUSTERS")
    print("=" * 60)

    cluster_profiles = df.groupby("cluster").agg({
        "customer_id": "count",
        "tenure": "mean",
        "monthly_charges": "mean",
        "num_tickets": "mean",
        "num_complaints": "mean",
        "late_payments": "mean",
        "mora_days": "mean",
        "churn_30": "mean",
        "churn_60": "mean",
    }).round(2)
    cluster_profiles.columns = [
        "N_Clientes", "Tenure_Prom", "ARPU_Prom", "Tickets_Prom", "Quejas_Prom",
        "Pagos_Tardios_Prom", "Mora_Dias_Prom", "Tasa_Churn_30", "Tasa_Churn_60",
    ]
    cluster_profiles["Porcentaje"] = (cluster_profiles["N_Clientes"] / len(df) * 100).round(1)
    print("\n" + cluster_profiles.to_string())

    cluster_names = _name_clusters(cluster_profiles)
    df["cluster_name"] = df["cluster"].map(cluster_names)

    print("\n\nNombres asignados a clusters:")
    for cluster, name in cluster_names.items():
        count = len(df[df["cluster"] == cluster])
        churn_rate = df[df["cluster"] == cluster]["churn_30"].mean() * 100
        print(f"  Cluster {cluster}: {name} ({count:,} clientes, {churn_rate:.1f}% churn)")

    return df, kmeans_final, cluster_profiles


def _name_clusters(cluster_profiles: pd.DataFrame) -> dict:
    cluster_names = {}
    for cluster in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster]
        if profile["Tenure_Prom"] > 30 and profile["Mora_Dias_Prom"] < 10:
            name = "Clientes Leales"
        elif profile["Mora_Dias_Prom"] > 20:
            name = "En Riesgo Financiero"
        elif profile["Quejas_Prom"] > 1.5:
            name = "Insatisfechos"
        elif profile["Tenure_Prom"] < 10:
            name = "Clientes Nuevos"
        else:
            name = f"Segmento {cluster}"
        cluster_names[cluster] = name
    return cluster_names
