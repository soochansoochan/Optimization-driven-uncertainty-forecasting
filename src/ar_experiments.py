# src/ar_experiments.py

import numpy as np
import pandas as pd

from .data_utils import add_penalty_cost, make_ar_design_for_hour
from .linear_forecasters import (
    fit_lad_linear_forecaster,
    fit_opt_driven_linear_forecaster,
)
from .metrics import predict_linear, evaluate_commitment


def run_ar_experiment_all_hours(
    df,
    n_lags=24,
    W1=1.0,
    W2=20.0,
    pred_bounds=(0.0, 1.0),
    solver_time_limit=5.0,
    verbose=False,
    include_baseline=True,
    include_proposed=True,
):
    """
    Run endogenous AR experiments across all daylight hour indices.

    Baseline AR minimizes LAD forecasting error. Proposed AR uses the
    paper-style optimization-driven objective, then both models are evaluated
    on the full test split.
    """

    H = df["hour_idx"].nunique()

    pred_rows = []
    coef_rows = []
    fit_rows = []

    for hour_idx in range(H):
        X_train, y_train, DP_train, RP_train, PC_train, train_meta = make_ar_design_for_hour(
            df,
            hour_idx=hour_idx,
            n_lags=n_lags,
            split="train",
        )

        X_test, y_test, DP_test, RP_test, PC_test, test_meta = make_ar_design_for_hour(
            df,
            hour_idx=hour_idx,
            n_lags=n_lags,
            split="test",
        )

        if include_baseline:
            baseline_fit = fit_lad_linear_forecaster(
                X_train,
                y_train,
                pred_bounds=pred_bounds,
                verbose=verbose,
            )

            pred_baseline = predict_linear(
                X_test,
                baseline_fit["alpha"],
                baseline_fit["beta"],
                clip_bounds=pred_bounds,
            )

            tmp = test_meta.copy()
            tmp["model"] = "Baseline AR"
            tmp["S_hat"] = pred_baseline
            pred_rows.append(tmp)

            coef_rows.append({
                "model": "Baseline AR",
                "hour_idx": hour_idx,
                "alpha": baseline_fit["alpha"],
                "objective": baseline_fit["objective"],
                "W1": None,
                "W2": None,
                "solver_status": baseline_fit["status"],
                "solver_runtime": None,
            })

        if include_proposed:
            proposed_fit = fit_opt_driven_linear_forecaster(
                X_train,
                y_train,
                DP_train,
                RP_train,
                PC_train,
                W1=W1,
                W2=W2,
                pred_bounds=pred_bounds,
                solver_time_limit=solver_time_limit,
                verbose=verbose,
            )

            pred_proposed = predict_linear(
                X_test,
                proposed_fit["alpha"],
                proposed_fit["beta"],
                clip_bounds=pred_bounds,
            )

            tmp = test_meta.copy()
            tmp["model"] = "Proposed AR"
            tmp["S_hat"] = pred_proposed
            pred_rows.append(tmp)

            coef_rows.append({
                "model": "Proposed AR",
                "hour_idx": hour_idx,
                "alpha": proposed_fit["alpha"],
                "objective": proposed_fit["objective"],
                "W1": W1,
                "W2": W2,
                "solver_status": proposed_fit["status"],
                "solver_runtime": proposed_fit["runtime"],
            })
            fit_rows.append({
                "model": "Proposed AR",
                "hour_idx": hour_idx,
                "solver_status": proposed_fit["status"],
                "solver_runtime": proposed_fit["runtime"],
            })

    pred_df = pd.concat(pred_rows, ignore_index=True)
    coef_df = pd.DataFrame(coef_rows)

    result_rows = []

    for model_name, g in pred_df.groupby("model"):
        metrics = evaluate_commitment(
            y_true=g["S"].to_numpy(),
            y_pred=g["S_hat"].to_numpy(),
            DP=g["DP"].to_numpy(),
            RP=g["RP"].to_numpy(),
            PC=g["PC"].to_numpy(),
            clip_bounds=pred_bounds,
        )

        row = {
            "model": model_name,
            "W1": W1 if model_name == "Proposed AR" else None,
            "W2": W2 if model_name == "Proposed AR" else None,
            "n_lags": n_lags,
            **metrics,
        }
        if model_name == "Proposed AR" and fit_rows:
            fit_df = pd.DataFrame(fit_rows)
            row["mean_solver_runtime"] = fit_df["solver_runtime"].mean()
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)

    return pred_df, result_df, coef_df


def sweep_ar_experiment(
    df,
    penalty_rates,
    weight_settings,
    add_penalty_cost_fn=None,
    n_lags=24,
    pred_bounds=(0.0, 1.0),
    solver_time_limit=5.0,
    verbose=False,
):
    """
    Sweep penalty-rate and W1/W2 settings for the AR experiments.
    """

    if add_penalty_cost_fn is None:
        add_penalty_cost_fn = add_penalty_cost

    all_results = []

    for penalty_rate in penalty_rates:
        df_pr = add_penalty_cost_fn(df, penalty_rate)

        baseline_added = False

        for setting_order, (W1, W2) in enumerate(weight_settings):
            _, result_df, _ = run_ar_experiment_all_hours(
                df_pr,
                n_lags=n_lags,
                W1=W1,
                W2=W2,
                pred_bounds=pred_bounds,
                solver_time_limit=solver_time_limit,
                verbose=verbose,
                include_baseline=not baseline_added,
                include_proposed=True,
            )

            result_df["penalty_rate"] = penalty_rate
            if "penalty_definition" in df_pr.columns:
                result_df["penalty_definition"] = df_pr["penalty_definition"].iloc[0]
            baseline_mask = result_df["model"].eq("Baseline AR")
            result_df.loc[baseline_mask, ["W1", "W2"]] = np.nan
            result_df.loc[~baseline_mask, "W1"] = W1
            result_df.loc[~baseline_mask, "W2"] = W2
            result_df["W_ratio"] = np.inf if W2 == 0 else W1 / W2
            result_df.loc[baseline_mask, "W_ratio"] = np.nan
            result_df["setting_order"] = setting_order
            result_df.loc[baseline_mask, "setting_order"] = -1

            all_results.append(result_df)
            baseline_added = True

    return pd.concat(all_results, ignore_index=True)
