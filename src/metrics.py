# src/metrics.py

import numpy as np


def predict_linear(X, alpha, beta, clip_bounds=None):
    """
    Linear prediction: y_hat = alpha + X @ beta
    """
    pred = np.asarray(alpha) + np.asarray(X) @ np.asarray(beta)

    if clip_bounds is not None:
        lb, ub = clip_bounds
        pred = np.clip(pred, lb, ub)

    return pred


def compute_commitment_profit(y_true, y_pred, DP, RP, PC, clip_bounds=(0.0, 1.0)):
    """
    Profit from day-ahead commitment following the paper formulation.

    If actual generation exceeds commitment, the surplus is sold at the
    real-time price. If commitment exceeds actual generation, the shortage
    pays the penalty cost.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    DP = np.asarray(DP, dtype=float)
    RP = np.asarray(RP, dtype=float)
    PC = np.asarray(PC, dtype=float)

    if clip_bounds is not None:
        lb, ub = clip_bounds
        y_pred = np.clip(y_pred, lb, ub)

    mismatch = y_true - y_pred
    surplus = np.maximum(0.0, mismatch)
    shortage = np.maximum(0.0, -mismatch)

    return DP * y_pred + RP * surplus - PC * shortage


def compute_oracle_profit(y_true, DP, RP, PC, clip_bounds=(0.0, 1.0)):
    """
    Eq. (13) perfect-information benchmark for optimality gap evaluation.

    This benchmark follows the paper's reported evaluation formula. It is not
    the objective used to fit the proposed forecasting model.
    """
    y_true = np.asarray(y_true, dtype=float)
    DP = np.asarray(DP, dtype=float)
    RP = np.asarray(RP, dtype=float)

    return np.maximum(DP, RP) * y_true


def evaluate_commitment(
    y_true,
    y_pred,
    DP,
    RP,
    PC,
    clip_bounds=(0.0, 1.0),
):
    """
    Forecasting accuracy and commitment-profit quality.

    y_true : actual renewable generation S_t
    y_pred : predicted generation / commitment S_hat_t
    DP     : day-ahead price
    RP     : real-time price
    PC     : penalty cost for shortage

    Returns:
    - nRMSE
    - optimality_gap_pct
    - profit
    - oracle_profit
    - mean_prediction
    - mean_shortage
    - mean_surplus
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    DP = np.asarray(DP, dtype=float)
    RP = np.asarray(RP, dtype=float)
    PC = np.asarray(PC, dtype=float)

    if clip_bounds is not None:
        lb, ub = clip_bounds
        y_pred = np.clip(y_pred, lb, ub)
        prediction_at_lower_rate = np.mean(np.isclose(y_pred, lb, atol=1e-6))
        prediction_at_upper_rate = np.mean(np.isclose(y_pred, ub, atol=1e-6))
    else:
        prediction_at_lower_rate = np.nan
        prediction_at_upper_rate = np.nan

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    nrmse = 100.0 * rmse / np.mean(y_true)

    imbalance = y_true - y_pred
    surplus = np.maximum(0.0, imbalance)
    shortage = np.maximum(0.0, -imbalance)

    realized_profit = compute_commitment_profit(
        y_true,
        y_pred,
        DP,
        RP,
        PC,
        clip_bounds=None,
    )

    oracle_profit = compute_oracle_profit(
        y_true,
        DP,
        RP,
        PC,
        clip_bounds=clip_bounds,
    )

    gap_abs = np.sum(oracle_profit) - np.sum(realized_profit)
    gap_pct = 100.0 * gap_abs / np.sum(oracle_profit)

    return {
        "nRMSE": nrmse,
        "optimality_gap_pct": gap_pct,
        "profit": np.sum(realized_profit),
        "oracle_profit": np.sum(oracle_profit),
        "mean_prediction": np.mean(y_pred),
        "mean_actual": np.mean(y_true),
        "mean_shortage": np.mean(shortage),
        "mean_surplus": np.mean(surplus),
        "prediction_at_lower_rate": prediction_at_lower_rate,
        "prediction_at_upper_rate": prediction_at_upper_rate,
    }
