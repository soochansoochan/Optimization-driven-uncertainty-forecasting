from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ar_experiments import sweep_ar_experiment
from src.data_utils import PAPER_MLR_DEACCUM_FEATURE_COLUMNS
from src.mlr_experiments import sweep_mlr_experiment


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "solar_miso_model_panel.csv"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
TABLE_PENALTY_RATE = 0.5
PENALTY_RATES = [0.0, 0.25, 0.5, 0.75, 1.0]

WEIGHT_SETTINGS = [
    (1.0, 20.0),
    (1.0, 10.0),
    (1.0, 5.0),
    (1.0, 2.0),
    (1.0, 1.0),
    (2.0, 1.0),
    (5.0, 1.0),
    (10.0, 1.0),
    (20.0, 1.0),
    (1.0, 0.0),
]

TABLE3_COLUMNS = [
    "penalty_rate",
    "penalty_definition",
    "model",
    "W1",
    "W2",
    "n_lags",
    "nRMSE",
    "optimality_gap_pct",
    "profit",
    "oracle_profit",
    "mean_prediction",
    "mean_actual",
    "mean_shortage",
    "mean_surplus",
    "prediction_at_lower_rate",
    "prediction_at_upper_rate",
    "setting_order",
    "mean_solver_runtime",
]

TABLE4_COLUMNS = [
    "feature_version",
    "penalty_rate",
    "penalty_definition",
    "model",
    "W1",
    "W2",
    "n_features",
    "selected_features",
    "selection_method",
    "standardized_features",
    "selection_threshold",
    "nRMSE",
    "optimality_gap_pct",
    "profit",
    "oracle_profit",
    "mean_prediction",
    "mean_actual",
    "mean_shortage",
    "mean_surplus",
    "prediction_at_lower_rate",
    "prediction_at_upper_rate",
    "setting_order",
    "solver_status",
    "solver_runtime",
]


def _ordered(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[col for col in columns if col in df.columns]]


def _warn_if_negative_gap(df: pd.DataFrame, label: str) -> None:
    if "optimality_gap_pct" not in df.columns:
        return
    bad = df[df["optimality_gap_pct"] < -1e-8]
    if bad.empty:
        return
    print(f"warning: negative optimality gap detected in {label}:")
    print(bad[["model", "W1", "W2", "penalty_rate", "optimality_gap_pct"]])


def _run_ar_weight_table(real_df: pd.DataFrame) -> pd.DataFrame:
    result = sweep_ar_experiment(
        real_df,
        penalty_rates=[TABLE_PENALTY_RATE],
        weight_settings=WEIGHT_SETTINGS,
        n_lags=24,
        pred_bounds=(0.0, 1.0),
        verbose=False,
    )
    result = _ordered(result, TABLE3_COLUMNS)
    return result.sort_values(["penalty_rate", "setting_order", "model"])


def _run_mlr_weight_table(real_df: pd.DataFrame) -> pd.DataFrame:
    result = sweep_mlr_experiment(
        real_df,
        penalty_rates=[TABLE_PENALTY_RATE],
        weight_settings=WEIGHT_SETTINGS,
        feature_cols=PAPER_MLR_DEACCUM_FEATURE_COLUMNS,
        selection_method="fixed",
        pred_bounds=(0.0, 1.0),
        verbose=False,
    )
    result["feature_version"] = "paper3_fixed_deaccum"
    result = _ordered(result, TABLE4_COLUMNS)
    return result.sort_values(["feature_version", "penalty_rate", "setting_order", "model"])


def _run_ar_penalty_sensitivity(real_df: pd.DataFrame) -> pd.DataFrame:
    result = sweep_ar_experiment(
        real_df,
        penalty_rates=PENALTY_RATES,
        weight_settings=[(1.0, 1.0)],
        n_lags=24,
        pred_bounds=(0.0, 1.0),
        verbose=False,
    )
    result = _ordered(result, TABLE3_COLUMNS)
    return result.sort_values(["penalty_rate", "setting_order", "model"])


def _run_mlr_penalty_sensitivity(real_df: pd.DataFrame) -> pd.DataFrame:
    result = sweep_mlr_experiment(
        real_df,
        penalty_rates=PENALTY_RATES,
        weight_settings=[(1.0, 1.0)],
        feature_cols=PAPER_MLR_DEACCUM_FEATURE_COLUMNS,
        selection_method="fixed",
        pred_bounds=(0.0, 1.0),
        verbose=False,
    )
    result["feature_version"] = "paper3_fixed_deaccum"
    result = _ordered(result, TABLE4_COLUMNS)
    return result.sort_values(["feature_version", "penalty_rate", "setting_order", "model"])


def main() -> None:
    real_df = pd.read_csv(DATA_PATH)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "table3_ar_results.csv": _run_ar_weight_table(real_df),
        "table4_mlr_results.csv": _run_mlr_weight_table(real_df),
        "ar_penalty_sensitivity.csv": _run_ar_penalty_sensitivity(real_df),
        "mlr_penalty_sensitivity.csv": _run_mlr_penalty_sensitivity(real_df),
    }

    for name, df in outputs.items():
        _warn_if_negative_gap(df, name)
        path = TABLE_DIR / name
        df.to_csv(path, index=False)
        print(f"saved: {path}")

    print("\nmain AR results")
    print(
        outputs["table3_ar_results.csv"][
            ["model", "W1", "W2", "nRMSE", "optimality_gap_pct", "mean_prediction"]
        ].to_string(index=False)
    )

    print("\nmain MLR results")
    print(
        outputs["table4_mlr_results.csv"][
            [
                "model",
                "W1",
                "W2",
                "n_features",
                "nRMSE",
                "optimality_gap_pct",
                "mean_prediction",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
