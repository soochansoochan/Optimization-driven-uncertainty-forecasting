import numpy as np
import pandas as pd

from .linear_forecasters import (
    fit_lad_linear_forecaster,
    fit_opt_driven_linear_forecaster,
)
from .data_utils import (
    add_penalty_cost,
    default_mlr_feature_columns,
    make_mlr_design,
)
from .feature_selection import backward_elimination
from .feature_selection import ols_p_values
from .metrics import evaluate_commitment, predict_linear


def run_mlr_experiment(
    df,
    W1=1.0,
    W2=20.0,
    feature_cols=None,
    selection_threshold=0.05,
    always_keep=(),
    selection_method="backward",
    standardize=False,
    pred_bounds=(0.0, 1.0),
    solver_time_limit=5.0,
    verbose=False,
    include_baseline=True,
    include_proposed=True,
):
    """
    Pooled MLR experiment using exogenous GEFCom predictors and paper-style
    calendar/time variables.
    """

    if feature_cols is None:
        feature_cols = default_mlr_feature_columns(df)
    feature_cols = list(feature_cols)

    X_select, y_select, _, _, _, _, _ = make_mlr_design(
        df,
        feature_cols=feature_cols,
        split="train",
        fit_scaler=True,
        standardize=standardize,
    )
    if selection_method == "backward":
        selection = backward_elimination(
            X_select,
            y_select,
            feature_cols=feature_cols,
            threshold=selection_threshold,
            always_keep=always_keep,
            min_features=max(1, len(always_keep or [])),
        )
        selected_features = selection["selected_features"]
    elif selection_method == "fixed":
        selected_features = feature_cols
        selection = {
            "selected_features": selected_features,
            "history": pd.DataFrame(),
            "final_stats": ols_p_values(X_select, y_select, selected_features),
        }
    else:
        raise ValueError(
            "selection_method must be either 'backward' or 'fixed'. "
            f"Got {selection_method!r}."
        )

    X_train, y_train, DP_train, RP_train, PC_train, train_meta, scaler = make_mlr_design(
        df,
        feature_cols=selected_features,
        split="train",
        fit_scaler=True,
        standardize=standardize,
    )
    X_test, y_test, DP_test, RP_test, PC_test, test_meta, _ = make_mlr_design(
        df,
        feature_cols=selected_features,
        split="test",
        scaler=scaler,
    )

    pred_rows = []
    coef_rows = []

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
        tmp["model"] = "Baseline MLR"
        tmp["S_hat"] = pred_baseline
        pred_rows.append(tmp)

        row = {
            "model": "Baseline MLR",
            "alpha": baseline_fit["alpha"],
            "objective": baseline_fit["objective"],
            "W1": None,
            "W2": None,
            "selected_features": ",".join(selected_features),
            "n_features": len(selected_features),
        }
        row.update(
            {
                f"beta_{feature}": baseline_fit["beta"][idx]
                for idx, feature in enumerate(selected_features)
            }
        )
        coef_rows.append(row)

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
        tmp["model"] = "Proposed MLR"
        tmp["S_hat"] = pred_proposed
        pred_rows.append(tmp)

        row = {
            "model": "Proposed MLR",
            "alpha": proposed_fit["alpha"],
            "objective": proposed_fit["objective"],
            "W1": W1,
            "W2": W2,
            "selected_features": ",".join(selected_features),
            "n_features": len(selected_features),
            "solver_status": proposed_fit["status"],
            "solver_runtime": proposed_fit["runtime"],
        }
        row.update(
            {
                f"beta_{feature}": proposed_fit["beta"][idx]
                for idx, feature in enumerate(selected_features)
            }
        )
        coef_rows.append(row)

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
            "W1": W1 if model_name == "Proposed MLR" else None,
            "W2": W2 if model_name == "Proposed MLR" else None,
            "selected_features": ",".join(selected_features),
            "n_features": len(selected_features),
            "selection_threshold": selection_threshold,
            "selection_method": selection_method,
            "standardized_features": bool(standardize),
            **metrics,
        }
        if model_name == "Proposed MLR":
            row["solver_status"] = proposed_fit["status"]
            row["solver_runtime"] = proposed_fit["runtime"]
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    selection_frames = []

    history = selection["history"].copy()
    if not history.empty:
        history["selection_table"] = "history"
        selection_frames.append(history)

    final_stats = selection["final_stats"].copy()
    final_stats["selection_table"] = "final_stats"
    selection_frames.append(final_stats)
    selection_df = pd.concat(selection_frames, ignore_index=True, sort=False)

    return pred_df, result_df, coef_df, selection_df


def sweep_mlr_experiment(
    df,
    penalty_rates,
    weight_settings,
    add_penalty_cost_fn=None,
    feature_cols=None,
    selection_threshold=0.05,
    always_keep=(),
    selection_method="backward",
    standardize=False,
    pred_bounds=(0.0, 1.0),
    solver_time_limit=5.0,
    verbose=False,
):
    if add_penalty_cost_fn is None:
        add_penalty_cost_fn = add_penalty_cost

    all_results = []

    for penalty_rate in penalty_rates:
        df_pr = add_penalty_cost_fn(df, penalty_rate)
        baseline_added = False

        for setting_order, (W1, W2) in enumerate(weight_settings):
            _, result_df, _, _ = run_mlr_experiment(
                df_pr,
                W1=W1,
                W2=W2,
                feature_cols=feature_cols,
                selection_threshold=selection_threshold,
                always_keep=always_keep,
                selection_method=selection_method,
                standardize=standardize,
                pred_bounds=pred_bounds,
                solver_time_limit=solver_time_limit,
                verbose=verbose,
                include_baseline=not baseline_added,
                include_proposed=True,
            )

            result_df["penalty_rate"] = penalty_rate
            if "penalty_definition" in df_pr.columns:
                result_df["penalty_definition"] = df_pr["penalty_definition"].iloc[0]
            baseline_mask = result_df["model"].eq("Baseline MLR")
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
