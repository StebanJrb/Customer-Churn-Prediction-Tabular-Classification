"""Synthetic Telmex Claro telecom dataset generator.

This is the single source of truth for the dataset used across Unidad 1, 2
and 3. All three original scripts hand-rolled their own copy of this exact
generation logic (same distributions, same risk-score formula) -- this module
replaces those three copies.
"""

import numpy as np
import pandas as pd

from churn.config import RANDOM_STATE


def generate_telecom_dataset(n_samples: int = 10000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a simulated telecom customer dataset with realistic churn logic.

    Churn is not random: a per-customer risk score is built from observable
    factors (contract type, tenure, complaints, tickets, payment behavior,
    consumption trend, seasonality) and converted to a probability via a
    sigmoid. churn_30 is a subset of churn_60 (a 30-day churner has
    necessarily also churned within 60 days).

    Parameters
    ----------
    n_samples : int
        Number of customers to generate.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per customer, with feature columns plus churn_30 / churn_60 targets.
    """
    np.random.seed(random_state)

    # Tenure (months): exponential -- most customers are recent, few are long-tenured.
    tenure = np.clip(np.random.exponential(scale=24, size=n_samples), 1, 72).astype(int)

    # Contract type is correlated with tenure: longer-tenured customers skew
    # toward longer commitments.
    contract_type = []
    for t in tenure:
        if t < 12:
            p = [0.7, 0.2, 0.1]
        elif t < 24:
            p = [0.4, 0.4, 0.2]
        else:
            p = [0.2, 0.3, 0.5]
        contract_type.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=p))
    contract_type = np.array(contract_type)

    # Monthly charges (ARPU, COP): normal, rounded to thousands, clipped to plan range.
    monthly_charges = np.round(np.random.normal(85000, 35000, n_samples), -3)
    monthly_charges = np.clip(monthly_charges, 30000, 200000)

    # Historical total billed = ARPU x tenure x noise factor (promos/adjustments).
    total_charges = monthly_charges * tenure * np.random.uniform(0.9, 1.1, n_samples)

    payment_method = np.random.choice(
        ["Electronic check", "Bank transfer", "Credit card", "Mailed check"],
        size=n_samples,
        p=[0.35, 0.25, 0.25, 0.15],
    )

    num_tickets = np.clip(np.random.poisson(lam=1.5, size=n_samples), 0, 15)
    num_complaints = np.clip(np.random.poisson(lam=0.8, size=n_samples), 0, 8)
    late_payments = np.clip(np.random.poisson(lam=1.2, size=n_samples), 0, 12)

    mora_days = np.where(
        late_payments > 0,
        np.random.exponential(scale=15, size=n_samples) * (late_payments / 3),
        0,
    )
    mora_days = np.clip(mora_days, 0, 90).astype(int)

    consumption_change = np.clip(np.random.normal(0, 15, n_samples), -50, 50)

    # Snapshot month, to capture the Jan/Feb seasonality bump in churn.
    month = np.random.choice(range(1, 13), size=n_samples)

    # --- Risk score -> churn probability ---
    risk_score = np.zeros(n_samples)
    risk_score += np.where(contract_type == "Month-to-month", 0.25, 0)
    risk_score += np.where(tenure < 6, 0.20, np.where(tenure < 12, 0.10, 0))
    risk_score += num_complaints * 0.08
    risk_score += num_tickets * 0.04
    risk_score += np.where(mora_days > 30, 0.15, np.where(mora_days > 15, 0.08, 0))
    risk_score += late_payments * 0.03
    risk_score += np.where(consumption_change < -20, 0.12, 0)
    risk_score += np.where(payment_method == "Electronic check", 0.05, 0)
    risk_score += np.where((month == 1) | (month == 2), 0.08, 0)

    churn_prob = 1 / (1 + np.exp(-5 * (risk_score - 0.35)))

    churn_60 = (np.random.random(n_samples) < churn_prob).astype(int)
    # A 60-day churner may already have churned within 30 days (~60% of them).
    churn_30 = np.where(churn_60 == 1, (np.random.random(n_samples) < 0.6).astype(int), 0)

    return pd.DataFrame({
        "customer_id": range(1, n_samples + 1),
        "tenure": tenure,
        "contract_type": contract_type,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "payment_method": payment_method,
        "num_tickets": num_tickets,
        "num_complaints": num_complaints,
        "late_payments": late_payments,
        "mora_days": mora_days,
        "consumption_change": consumption_change,
        "month": month,
        "churn_30": churn_30,
        "churn_60": churn_60,
    })
