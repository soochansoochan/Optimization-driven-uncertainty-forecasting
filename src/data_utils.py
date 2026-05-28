# src/data_utils.py

import numpy as np
import pandas as pd


REQUIRED_COLUMNS_AR = [
    "day_idx",
    "hour",
    "hour_idx",
    "t_step",
    "split",
    "S",
    "DP",
    "RP",
]

PREDICTOR_COLUMNS = [
    "VAR78",
    "VAR79",
    "VAR134",
    "VAR157",
    "VAR164",
    "VAR165",
    "VAR166",
    "VAR167",
    "VAR169",
    "VAR175",
    "VAR178",
    "VAR228",
]

ACCUMULATED_PREDICTOR_COLUMNS = ["VAR169", "VAR175", "VAR178", "VAR228"]
DEACCUMULATED_PREDICTOR_MAP = {
    col: f"{col}_deaccum" for col in ACCUMULATED_PREDICTOR_COLUMNS
}

DEACCUMULATED_ONLY_PREDICTOR_COLUMNS = list(DEACCUMULATED_PREDICTOR_MAP.values())
DEACCUMULATED_PREDICTOR_COLUMNS = [
    DEACCUMULATED_PREDICTOR_MAP.get(col, col) for col in PREDICTOR_COLUMNS
]

MLR_FEATURE_COLUMNS = ["month", "hour", *PREDICTOR_COLUMNS]
PAPER_MLR_FEATURE_COLUMNS = ["hour", "VAR169", "VAR178"]
MLR_DEACCUM_FEATURE_COLUMNS = ["month", "hour", *DEACCUMULATED_PREDICTOR_COLUMNS]
PAPER_MLR_DEACCUM_FEATURE_COLUMNS = [
    "hour",
    DEACCUMULATED_PREDICTOR_MAP["VAR169"],
    DEACCUMULATED_PREDICTOR_MAP["VAR178"],
]


def add_deaccumulated_predictors(
    df,
    columns=ACCUMULATED_PREDICTOR_COLUMNS,
    group_cols=("ZONEID",),
    sort_cols=None,
    reset_fraction=0.5,
):
    """
    Add hourly increments for accumulated GEFCom weather predictors.

    The ECMWF radiation/precipitation fields in GEFCom are accumulated over
    forecast steps. A large drop indicates a reset to a new accumulation cycle;
    small negative differences are clipped to zero to avoid turning rounding
    noise into a full reset.
    """

    out = df.copy()
    columns = [col for col in columns if col in out.columns]
    if not columns:
        return out

    if sort_cols is None:
        for candidate in [
            ["solar_timestamp_raw"],
            ["TIMESTAMP"],
            ["t_step"],
            ["day_idx", "hour_idx"],
        ]:
            if all(col in out.columns for col in candidate):
                sort_cols = candidate
                break
        else:
            raise ValueError("Could not infer sort columns for deaccumulation.")

    sort_cols = list(sort_cols)
    group_cols = [col for col in group_cols if col in out.columns]
    order_cols = [*group_cols, *sort_cols]
    work = out.sort_values(order_cols).copy()

    if group_cols:
        grouped = work.groupby(group_cols, sort=False)
    else:
        grouped = [(None, work)]

    for col in columns:
        if group_cols:
            prev = grouped[col].shift(1)
        else:
            prev = work[col].shift(1)

        current = pd.to_numeric(work[col], errors="coerce")
        prev = pd.to_numeric(prev, errors="coerce")
        diff = current - prev
        first = prev.isna()
        reset = prev.notna() & (current < prev * reset_fraction)

        increment = diff.mask(first | reset, current)
        increment = increment.clip(lower=0.0).fillna(0.0)
        out.loc[work.index, DEACCUMULATED_PREDICTOR_MAP[col]] = increment

    return out


def validate_ar_dataframe(df):
    missing = [col for col in REQUIRED_COLUMNS_AR if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["S"].isna().any():
        raise ValueError("Column S contains NaN.")
    if df["DP"].isna().any():
        raise ValueError("Column DP contains NaN.")
    if df["RP"].isna().any():
        raise ValueError("Column RP contains NaN.")


def default_mlr_feature_columns(df):
    return [col for col in MLR_FEATURE_COLUMNS if col in df.columns]


def validate_mlr_dataframe(df, feature_cols=None):
    validate_ar_dataframe(df)

    if feature_cols is None:
        feature_cols = default_mlr_feature_columns(df)

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required MLR columns: {missing}")

    if not feature_cols:
        raise ValueError("No MLR feature columns were provided or found.")

    for col in feature_cols:
        if df[col].isna().any():
            raise ValueError(f"MLR feature column {col} contains NaN.")


def add_penalty_cost(df, penalty_rate):
    """
    Paper-style shortage penalty parameter:
        PC_t = (1 + penalty_rate) * DP_t

    The realized shortage penalty in the model and metrics is PC_t multiplied
    by the shortage amount.
    """
    out = df.copy()
    out["penalty_rate"] = float(penalty_rate)
    out["PC"] = (1.0 + out["penalty_rate"]) * out["DP"]
    out["penalty_definition"] = "PC=(1+r)DP"
    return out


def fit_feature_scaler(meta, feature_cols, standardize=False):
    feature_frame = meta[feature_cols].astype(float)
    if standardize:
        mean = feature_frame.mean()
        std = feature_frame.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    else:
        mean = pd.Series(0.0, index=feature_cols)
        std = pd.Series(1.0, index=feature_cols)
    return {
        "feature_cols": list(feature_cols),
        "mean": mean,
        "std": std,
        "standardize": bool(standardize),
    }


def transform_features(meta, scaler):
    feature_cols = scaler["feature_cols"]
    feature_frame = meta[feature_cols].astype(float)
    return ((feature_frame - scaler["mean"]) / scaler["std"]).to_numpy(dtype=float)


def make_mlr_design(
    df,
    feature_cols=None,
    split="train",
    scaler=None,
    fit_scaler=False,
    standardize=False,
):
    """
    Pooled MLR design matrix for all daylight hours.

    The paper-style time variables are month and hour.
    """

    if feature_cols is None:
        feature_cols = default_mlr_feature_columns(df)
    feature_cols = list(feature_cols)

    validate_mlr_dataframe(df, feature_cols)

    if "PC" not in df.columns:
        raise ValueError("Column PC is required. Run add_penalty_cost() first.")

    out = df[df["split"] == split].sort_values("t_step").copy()
    if out.empty:
        raise ValueError(f"No rows found for split={split!r}.")

    if fit_scaler:
        scaler = fit_feature_scaler(out, feature_cols, standardize=standardize)
    elif scaler is None:
        raise ValueError("Provide scaler or set fit_scaler=True.")

    X = transform_features(out, scaler)
    y = out["S"].to_numpy(dtype=float)
    DP = out["DP"].to_numpy(dtype=float)
    RP = out["RP"].to_numpy(dtype=float)
    PC = out["PC"].to_numpy(dtype=float)

    return X, y, DP, RP, PC, out, scaler


def make_ar_design_for_hour(
    df,
    hour_idx,
    n_lags=24,
    split="train",
):
    """
    Build the direct multi-step AR design matrix for one daylight hour.

    Target:
        S at target day and hour_idx

    Features:
        previous n_lags observations before the start of the target day.
    """

    validate_ar_dataframe(df)

    df = df.sort_values("t_step").reset_index(drop=True)

    if "PC" not in df.columns:
        raise ValueError("Column PC is required. Run add_penalty_cost() first.")

    H = df["hour_idx"].nunique()

    s_by_t = df.set_index("t_step")["S"].to_dict()

    target_rows = df[
        (df["hour_idx"] == hour_idx)
        & (df["split"] == split)
    ].copy()

    records = []

    for _, row in target_rows.iterrows():
        day_idx = int(row["day_idx"])
        day_start_t = day_idx * H

        lag_values = []
        valid = True

        for l in range(1, n_lags + 1):
            lag_t = day_start_t - l

            if lag_t not in s_by_t:
                valid = False
                break

            lag_values.append(s_by_t[lag_t])

        if not valid:
            continue

        rec = {
            "day_idx": day_idx,
            "hour": int(row["hour"]),
            "hour_idx": int(row["hour_idx"]),
            "t_step": int(row["t_step"]),
            "split": split,
            "S": float(row["S"]),
            "DP": float(row["DP"]),
            "RP": float(row["RP"]),
            "PC": float(row["PC"]),
        }

        for l, val in enumerate(lag_values, start=1):
            rec[f"lag_{l}"] = float(val)

        records.append(rec)

    out = pd.DataFrame(records)

    if out.empty:
        raise ValueError(
            f"No valid samples for hour_idx={hour_idx}, split={split}. "
            f"Check n_lags or data range."
        )

    lag_cols = [f"lag_{l}" for l in range(1, n_lags + 1)]

    X = out[lag_cols].to_numpy(dtype=float)
    y = out["S"].to_numpy(dtype=float)
    DP = out["DP"].to_numpy(dtype=float)
    RP = out["RP"].to_numpy(dtype=float)
    PC = out["PC"].to_numpy(dtype=float)

    return X, y, DP, RP, PC, out
