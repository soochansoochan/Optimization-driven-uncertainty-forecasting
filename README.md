# Optimization-Driven Uncertainty Forecasting

Python/Gurobi implementation of
[Optimization-driven uncertainty forecasting: Application to day-ahead commitment with renewable energy resources](https://doi.org/10.1016/j.apenergy.2022.119929)
by Karimi and Kwon (2022).

The repository contains Python/Gurobi code for AR and MLR experiments,
including result tables and figures.

## Requirements

- Python 3.10+
- Gurobi with a valid license
- Python dependencies listed in `requirements.txt`

## Repository Structure

```text
data/processed/
  solar_miso_model_panel.csv        Processed solar/weather/price panel

src/
  linear_forecasters.py             Baseline LAD and proposed LP models
  ar_experiments.py                 AR experiment runner
  mlr_experiments.py                MLR experiment runner
  data_utils.py                     Feature creation and penalty cost utilities
  metrics.py                        nRMSE, profit, and optimality gap metrics
  feature_selection.py              OLS p-value helper for comparison runs

scripts/
  run_paper_experiments.py          Generate AR/MLR result tables
  make_result_figures.py            Generate result figures

results/tables/                     Result CSV and Markdown tables
results/figures/                    Result figures
```

## Main Implementation Settings

```text
Solar generation scale : S_t in [0, 1]
Daylight hours         : 09:00-20:00
Train/test split       : first 264 days train, last 100 days test
Penalty cost           : PC_t = (1 + penalty_rate) * DP_t
Main penalty rate      : 50%
Prediction bound       : 0 <= S_hat_t <= 1
Final MLR features     : hour, VAR169_deaccum, VAR178_deaccum
```

## Run

```powershell
git clone https://github.com/soochansoochan/Optimization-driven-uncertainty-forecasting.git
cd Optimization-driven-uncertainty-forecasting

pip install -r requirements.txt

python scripts\run_paper_experiments.py
python scripts\make_result_figures.py
```

## Main Outputs

```text
results/tables/table3_ar_results.csv
results/tables/table4_mlr_results.csv
results/tables/ar_penalty_sensitivity.csv
results/tables/mlr_penalty_sensitivity.csv

results/figures/fig3_ar_weight_sensitivity.png
results/figures/fig4_ar_prediction_window.png
results/figures/fig5_ar_penalty_sensitivity.png
results/figures/fig6_mlr_weight_sensitivity.png
results/figures/fig7_mlr_prediction_window.png
results/figures/fig8_mlr_penalty_sensitivity.png
```

## Data

The repository includes the processed panel used by the scripts. Raw GEFCom and
MISO source files are not included.

The solar/weather series and price series are matched by day index and hour.
