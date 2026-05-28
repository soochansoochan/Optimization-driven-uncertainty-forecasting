# src/linear_forecasters.py

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def _check_xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    return X, y


def fit_lad_linear_forecaster(
    X,
    y,
    pred_bounds=(0.0, 1.0),
    verbose=False,
):
    """
    Baseline linear forecasting model trained by LAD loss.

    Minimize:
        sum_i |y_i - (alpha + beta^T x_i)|

    This corresponds to the conventional forecasting model
    that minimizes prediction error only.
    """

    X, y = _check_xy(X, y)

    n_samples, n_features = X.shape
    I = range(n_samples)
    K = range(n_features)

    model = gp.Model("baseline_lad_linear_forecaster")

    if not verbose:
        model.setParam("OutputFlag", 0)

    alpha = model.addVar(lb=-GRB.INFINITY, name="alpha")
    beta = model.addVars(K, lb=-GRB.INFINITY, name="beta")

    abs_error = model.addVars(I, lb=0.0, name="abs_error")

    for i in I:
        pred_i = alpha + gp.quicksum(beta[k] * X[i, k] for k in K)

        if pred_bounds is not None:
            lb, ub = pred_bounds
            model.addConstr(pred_i >= lb, name=f"pred_lb_{i}")
            model.addConstr(pred_i <= ub, name=f"pred_ub_{i}")

        error_i = y[i] - pred_i

        model.addConstr(abs_error[i] >= error_i, name=f"abs_pos_{i}")
        model.addConstr(abs_error[i] >= -error_i, name=f"abs_neg_{i}")

    model.setObjective(
        gp.quicksum(abs_error[i] for i in I),
        GRB.MINIMIZE,
    )

    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Optimization failed. Gurobi status: {model.status}")

    alpha_hat = alpha.X
    beta_hat = np.array([beta[k].X for k in K])

    return {
        "alpha": alpha_hat,
        "beta": beta_hat,
        "objective": model.ObjVal,
        "status": model.status,
    }


def fit_opt_driven_linear_forecaster(
    X,
    y,
    DP,
    RP,
    PC,
    W1=1.0,
    W2=20.0,
    pred_bounds=(0.0, 1.0),
    solver_time_limit=None,
    verbose=False,
):
    """
    Paper-style optimization-driven linear forecasting model.

    The same solver is used for endogenous AR and exogenous MLR designs. The
    matrix X determines whether the features are lagged generation or weather
    predictors.

    Minimize:
        sum_i [
            W1 * profit_loss_i
            + W2 * absolute_forecasting_error_i
        ]

    where predicted generation is interpreted as day-ahead commitment. The
    surplus/shortage variables follow Eq. (9)/(10) in the paper:
        y_i = S_i - x_i
        y_i = surplus_i - shortage_i
        surplus_i <= S_i
    """

    X, y = _check_xy(X, y)
    DP = np.asarray(DP, dtype=float)
    RP = np.asarray(RP, dtype=float)
    PC = np.asarray(PC, dtype=float)

    n_samples, n_features = X.shape

    if not (len(DP) == len(RP) == len(PC) == n_samples):
        raise ValueError("DP, RP, PC must have the same length as y.")

    I = range(n_samples)
    K = range(n_features)

    model = gp.Model("opt_driven_linear_forecaster")

    if not verbose:
        model.setParam("OutputFlag", 0)
    if solver_time_limit is not None:
        model.setParam("TimeLimit", float(solver_time_limit))

    alpha = model.addVar(lb=-GRB.INFINITY, name="alpha")
    beta = model.addVars(K, lb=-GRB.INFINITY, name="beta")

    abs_error = model.addVars(I, lb=0.0, name="abs_error")
    mismatch = model.addVars(I, lb=-GRB.INFINITY, name="mismatch")
    surplus = model.addVars(I, lb=0.0, name="surplus")
    shortage = model.addVars(I, lb=0.0, name="shortage")

    obj_terms = []

    for i in I:
        pred_i = alpha + gp.quicksum(beta[k] * X[i, k] for k in K)

        if pred_bounds is not None:
            lb, ub = pred_bounds
            model.addConstr(pred_i >= lb, name=f"pred_lb_{i}")
            model.addConstr(pred_i <= ub, name=f"pred_ub_{i}")

        error_i = y[i] - pred_i

        model.addConstr(abs_error[i] >= error_i, name=f"abs_pos_{i}")
        model.addConstr(abs_error[i] >= -error_i, name=f"abs_neg_{i}")

        model.addConstr(mismatch[i] == y[i] - pred_i, name=f"mismatch_def_{i}")
        model.addConstr(mismatch[i] == surplus[i] - shortage[i], name=f"surplus_shortage_{i}")
        model.addConstr(surplus[i] <= y[i], name=f"surplus_limit_{i}")

        actual_profit_i = DP[i] * pred_i + RP[i] * surplus[i] - PC[i] * shortage[i]
        perfect_profit_i = DP[i] * y[i]
        profit_loss_i = perfect_profit_i - actual_profit_i

        obj_terms.append(W1 * profit_loss_i + W2 * abs_error[i])

    model.setObjective(
        gp.quicksum(obj_terms),
        GRB.MINIMIZE,
    )

    model.optimize()

    acceptable_statuses = {GRB.OPTIMAL}
    if solver_time_limit is not None:
        acceptable_statuses.add(GRB.TIME_LIMIT)

    if model.status not in acceptable_statuses or model.SolCount == 0:
        raise RuntimeError(f"Optimization failed. Gurobi status: {model.status}")

    alpha_hat = alpha.X
    beta_hat = np.array([beta[k].X for k in K])

    return {
        "alpha": alpha_hat,
        "beta": beta_hat,
        "objective": model.ObjVal,
        "status": model.status,
        "runtime": model.Runtime,
        "sol_count": model.SolCount,
    }
