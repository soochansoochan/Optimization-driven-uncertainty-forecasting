import math

import numpy as np
import pandas as pd


def _two_sided_p_value(t_value, dof):
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t_value), dof))
    except Exception:
        return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def ols_p_values(X, y, feature_cols):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    feature_cols = list(feature_cols)

    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    if X.shape[1] != len(feature_cols):
        raise ValueError("X column count must match feature_cols.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    design = np.column_stack([np.ones(X.shape[0]), X])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ coef

    dof = design.shape[0] - design.shape[1]
    if dof <= 0:
        raise ValueError("Not enough samples to compute OLS p-values.")

    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    rows = []
    for idx, name in enumerate(["intercept", *feature_cols]):
        if se[idx] <= 0.0:
            t_value = np.nan
            p_value = np.nan
        else:
            t_value = coef[idx] / se[idx]
            p_value = _two_sided_p_value(t_value, dof)
        rows.append(
            {
                "feature": name,
                "coef": coef[idx],
                "std_error": se[idx],
                "t_value": t_value,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows)


def backward_elimination(
    X,
    y,
    feature_cols,
    threshold=0.05,
    always_keep=(),
    min_features=1,
):
    """
    Backward elimination using OLS p-values on the training design matrix.

    p-values use scipy's Student-t distribution when available. Without scipy,
    the normal approximation is used, which is accurate for this dataset's
    thousands of training observations.
    """

    selected = list(feature_cols)
    always_keep = set(always_keep or [])
    always_keep = always_keep & set(selected)
    history = []

    while True:
        stats = ols_p_values(
            np.asarray(X)[:, [feature_cols.index(col) for col in selected]],
            y,
            selected,
        )
        candidates = stats[
            stats["feature"].ne("intercept")
            & ~stats["feature"].isin(always_keep)
        ].copy()

        if candidates.empty or len(selected) <= min_features:
            break

        candidates["sort_p"] = candidates["p_value"].fillna(np.inf)
        worst = candidates.sort_values("sort_p", ascending=False).iloc[0]
        worst_feature = str(worst["feature"])
        worst_p = float(worst["sort_p"])

        if worst_p <= threshold:
            break

        history.append(
            {
                "step": len(history),
                "removed_feature": worst_feature,
                "removed_p_value": worst_p,
                "selected_before": ",".join(selected),
            }
        )

        selected.remove(worst_feature)
        history[-1]["selected_after"] = ",".join(selected)

    final_stats = ols_p_values(
        np.asarray(X)[:, [feature_cols.index(col) for col in selected]],
        y,
        selected,
    )

    return {
        "selected_features": selected,
        "history": pd.DataFrame(history),
        "final_stats": final_stats,
    }
