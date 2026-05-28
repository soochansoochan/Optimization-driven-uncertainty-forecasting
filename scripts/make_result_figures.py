from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ar_experiments import run_ar_experiment_all_hours
from src.data_utils import add_penalty_cost, PAPER_MLR_DEACCUM_FEATURE_COLUMNS
from src.mlr_experiments import run_mlr_experiment


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "solar_miso_model_panel.csv"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
REPORT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
PENALTY_RATE = 0.5
PREDICTION_W1 = 1.0
PREDICTION_W2 = 1.0
PREDICTION_WINDOW_DAYS = 7
SOLAR_CAPACITY_MW = 30.0

PAPER_BLUE = "#4472C4"
PAPER_ORANGE = "#ED7D31"
PAPER_RED = "#FF0000"
PAPER_GRAY = "#A5A5A5"
GRID_GRAY = "#D9D9D9"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def _percent_axis(value, _position):
    return f"{value:.0f}%"


def _setting_label(row: pd.Series) -> str:
    if pd.isna(row["W1"]):
        return "Baseline"
    return f"{row['W1']:g}:{row['W2']:g}"


def plot_weight_sensitivity(df: pd.DataFrame, title: str, path: Path) -> None:
    baseline = df[df["model"].str.contains("Baseline", na=False)]
    proposed = df[df["model"].str.contains("Proposed", na=False)].copy()
    proposed = proposed.sort_values("setting_order")
    proposed["setting"] = proposed.apply(_setting_label, axis=1)
    x_labels = proposed["setting"].tolist()
    nrmse = proposed["nRMSE"].tolist()
    og = proposed["optimality_gap_pct"].tolist()
    if not baseline.empty:
        baseline_label = "AR" if "AR" in baseline["model"].iloc[0] else "MLR"
        x_labels = [baseline_label, *x_labels]
        nrmse = [baseline["nRMSE"].iloc[0], *nrmse]
        og = [baseline["optimality_gap_pct"].iloc[0], *og]

    x = range(len(x_labels))
    fig, ax1 = plt.subplots(figsize=(6.4, 4.1))
    ax2 = ax1.twinx()

    line1 = ax1.plot(x, nrmse, color=PAPER_ORANGE, linewidth=1.8, label="nRMSE")
    line2 = ax2.plot(x, og, color=PAPER_BLUE, linewidth=1.8, label="Optimality Gap")

    ax1.set_title(title, fontsize=11, weight="bold")
    ax1.set_xlabel("W1/W2", weight="bold")
    ax1.set_ylabel("nRMSE", weight="bold")
    ax2.set_ylabel("Optimality Gap", weight="bold")
    ax1.yaxis.set_major_formatter(FuncFormatter(_percent_axis))
    ax2.yaxis.set_major_formatter(FuncFormatter(_percent_axis))
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(x_labels, fontsize=9)
    ax1.grid(True, axis="y", color=GRID_GRAY, linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(line1 + line2, [line.get_label() for line in line1 + line2], loc="upper center", ncol=2, frameon=True, fancybox=False, edgecolor="0.4")
    for spine in [*ax1.spines.values(), *ax2.spines.values()]:
        spine.set_color("0.25")
        spine.set_linewidth(1.0)
    _save(fig, path)


def plot_penalty_sensitivity(df: pd.DataFrame, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    for model, group in df.sort_values("penalty_rate").groupby("model"):
        is_baseline = "Baseline" in model
        color = PAPER_BLUE if is_baseline else PAPER_ORANGE
        label = "AR" if model == "Baseline AR" else "MLR" if model == "Baseline MLR" else "Proposed Model"
        x = group["penalty_rate"] * 100.0
        axes[0].plot(
            x,
            group["nRMSE"],
            color=color,
            linewidth=1.8,
            label=label,
        )
        axes[1].plot(
            x,
            group["optimality_gap_pct"],
            color=color,
            linewidth=1.8,
            label=label,
        )

    axes[0].set_ylabel("nRMSE", weight="bold")
    axes[1].set_ylabel("Optimality Gap", weight="bold")
    for ax in axes:
        ax.set_xlabel("Penalty Cost Rate", weight="bold")
        ax.xaxis.set_major_formatter(FuncFormatter(_percent_axis))
        ax.yaxis.set_major_formatter(FuncFormatter(_percent_axis))
        ax.grid(True, axis="y", color=GRID_GRAY, linewidth=0.8)
        ax.legend(loc="upper center", ncol=2, frameon=True, fancybox=False, edgecolor="0.4")
        for spine in ax.spines.values():
            spine.set_color("0.25")
            spine.set_linewidth(1.0)
    fig.suptitle(title)
    _save(fig, path)


def _prediction_frame(pred_df: pd.DataFrame, days: int) -> pd.DataFrame:
    meta_cols = ["t_step", "day_idx", "hour", "S"]
    extra_cols = [col for col in ["price_date", "solar_date_local"] if col in pred_df.columns]
    actual = (
        pred_df[[*meta_cols, *extra_cols]]
        .drop_duplicates("t_step")
        .sort_values("t_step")
    )
    pred = pred_df.pivot_table(index="t_step", columns="model", values="S_hat")
    daylight = actual.merge(pred, on="t_step", how="left")

    selected_days = sorted(daylight["day_idx"].unique())[:days]
    daylight = daylight[daylight["day_idx"].isin(selected_days)].copy()
    model_cols = [
        col
        for col in ["Baseline AR", "Proposed AR", "Baseline MLR", "Proposed MLR"]
        if col in daylight.columns
    ]

    records = []
    for day_order, day_idx in enumerate(selected_days):
        day_rows = daylight[daylight["day_idx"].eq(day_idx)].set_index("hour")
        for hour in range(24):
            rec = {
                "time_h": day_order * 24 + hour + 1,
                "day_idx": int(day_idx),
                "hour": hour,
                "S": 0.0,
            }
            if not day_rows.empty:
                for col in extra_cols:
                    rec[col] = day_rows[col].iloc[0]
            if hour in day_rows.index:
                row = day_rows.loc[hour]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                rec["S"] = float(row["S"])
                for col in model_cols:
                    rec[col] = float(row[col])
            else:
                for col in model_cols:
                    rec[col] = 0.0
            records.append(rec)

    return pd.DataFrame(records)


def plot_prediction_window(frame: pd.DataFrame, title: str, path: Path) -> None:
    x = frame["time_h"]
    fig, ax = plt.subplots(figsize=(10.4, 4.1))
    ax.plot(x, frame["S"] * SOLAR_CAPACITY_MW, color=PAPER_BLUE, linewidth=1.9, label="Actual")
    for column, color, label in [
        ("Proposed AR", PAPER_RED, "Proposed Model"),
        ("Baseline AR", PAPER_GRAY, "AR"),
        ("Proposed MLR", PAPER_RED, "Proposed Model"),
        ("Baseline MLR", PAPER_GRAY, "MLR"),
    ]:
        if column in frame.columns:
            ax.plot(x, frame[column] * SOLAR_CAPACITY_MW, linewidth=1.8, label=label, color=color)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("Time (h)", weight="bold")
    ax.set_ylabel("Solar Power (MW)", weight="bold")
    ax.set_ylim(0.0, SOLAR_CAPACITY_MW)
    ax.grid(True, axis="y", color=GRID_GRAY, linewidth=0.8)
    ax.legend(loc="upper center", ncol=3, frameon=True, fancybox=False, edgecolor="0.4")
    for spine in ax.spines.values():
        spine.set_color("0.25")
        spine.set_linewidth(1.0)
    _save(fig, path)


def _format_table_frame(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    out = df.copy().sort_values("setting_order")
    out["Models"] = out["model"].map(
        {
            f"Baseline {model_label}": f"{model_label} model",
            f"Proposed {model_label}": "Proposed model",
        }
    ).fillna(out["model"])
    out["W1"] = out["W1"].apply(lambda x: "-" if pd.isna(x) else f"{x:g}")
    out["W2"] = out["W2"].apply(lambda x: "-" if pd.isna(x) else f"{x:g}")
    out["nRMSE"] = out["nRMSE"].map(lambda x: f"{x:.2f}%")
    out["Optimality gap (%)"] = out["optimality_gap_pct"].map(lambda x: f"{x:.2f}%")
    return out[["Models", "W1", "W2", "nRMSE", "Optimality gap (%)"]]


def save_markdown_table(table: pd.DataFrame, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"{title}\n\n{table.to_markdown(index=False)}\n"
    path.write_text(text, encoding="utf-8")
    print(f"saved: {path}")


def plot_table(table: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 0.48 * len(table) + 1.35))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10, weight="bold", pad=8)
    rendered = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.92],
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8.5)
    rendered.scale(1.0, 1.1)
    for (row, col), cell in rendered.get_celld().items():
        cell.set_edgecolor("0.35")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F2F2F2")
        if row in (1, 2) and col in (3, 4):
            cell.set_facecolor("#FFF176")
            cell.set_text_props(weight="bold")
    _save(fig, path)


def make_tables() -> None:
    ar_weight = pd.read_csv(TABLE_DIR / "table3_ar_results.csv")
    mlr_weight = pd.read_csv(TABLE_DIR / "table4_mlr_results.csv")
    table3 = _format_table_frame(ar_weight, "AR")
    table4 = _format_table_frame(mlr_weight, "MLR")
    save_markdown_table(
        table3,
        "Table 3. Comparison between AR and proposed model when penalty cost rate is 50%.",
        REPORT_TABLE_DIR / "table3_ar_results.md",
    )
    save_markdown_table(
        table4,
        "Table 4. Comparison between MLR and proposed model when penalty cost rate is 50%.",
        REPORT_TABLE_DIR / "table4_mlr_results.md",
    )
    plot_table(
        table3,
        "Table 3\nComparison between AR and proposed model when penalty cost rate is 50%.",
        FIGURE_DIR / "table3_ar_results.png",
    )
    plot_table(
        table4,
        "Table 4\nComparison between MLR and proposed model when penalty cost rate is 50%.",
        FIGURE_DIR / "table4_mlr_results.png",
    )


def make_prediction_figures() -> None:
    real_df = pd.read_csv(DATA_PATH)
    df = add_penalty_cost(real_df, PENALTY_RATE)

    ar_pred, _, _ = run_ar_experiment_all_hours(
        df,
        n_lags=24,
        W1=PREDICTION_W1,
        W2=PREDICTION_W2,
        pred_bounds=(0.0, 1.0),
        verbose=False,
        include_baseline=True,
        include_proposed=True,
    )
    ar_frame = _prediction_frame(ar_pred, PREDICTION_WINDOW_DAYS)
    ar_frame.to_csv(FIGURE_DIR / "fig4_ar_prediction_window_data.csv", index=False)
    plot_prediction_window(
        ar_frame,
        "Fig. 4. Actual vs AR forecasts",
        FIGURE_DIR / "fig4_ar_prediction_window.png",
    )

    mlr_pred, _, _, _ = run_mlr_experiment(
        df,
        W1=PREDICTION_W1,
        W2=PREDICTION_W2,
        feature_cols=PAPER_MLR_DEACCUM_FEATURE_COLUMNS,
        selection_method="fixed",
        pred_bounds=(0.0, 1.0),
        verbose=False,
        include_baseline=True,
        include_proposed=True,
    )
    mlr_frame = _prediction_frame(mlr_pred, PREDICTION_WINDOW_DAYS)
    mlr_frame.to_csv(FIGURE_DIR / "fig7_mlr_prediction_window_data.csv", index=False)
    plot_prediction_window(
        mlr_frame,
        "Fig. 7. Actual vs MLR forecasts",
        FIGURE_DIR / "fig7_mlr_prediction_window.png",
    )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    ar_weight = pd.read_csv(TABLE_DIR / "table3_ar_results.csv")
    mlr_weight = pd.read_csv(TABLE_DIR / "table4_mlr_results.csv")
    ar_penalty = pd.read_csv(TABLE_DIR / "ar_penalty_sensitivity.csv")
    mlr_penalty = pd.read_csv(TABLE_DIR / "mlr_penalty_sensitivity.csv")

    plot_weight_sensitivity(
        ar_weight,
        "nRMSE and Optimality Gap",
        FIGURE_DIR / "fig3_ar_weight_sensitivity.png",
    )
    plot_penalty_sensitivity(
        ar_penalty,
        "Comparison of forecasting error and optimality gap of AR and proposed model",
        FIGURE_DIR / "fig5_ar_penalty_sensitivity.png",
    )
    plot_weight_sensitivity(
        mlr_weight,
        "nRMSE and Optimality Gap",
        FIGURE_DIR / "fig6_mlr_weight_sensitivity.png",
    )
    plot_penalty_sensitivity(
        mlr_penalty,
        "Comparison of forecasting error and optimality gap of MLR and proposed model",
        FIGURE_DIR / "fig8_mlr_penalty_sensitivity.png",
    )
    make_prediction_figures()
    make_tables()


if __name__ == "__main__":
    main()
