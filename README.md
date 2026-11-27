**Adaptive Ensemble**

**Introduction**

The adaptive ensemble approach combines epidemic projections from the Flu Scenario Modeling Hub across all defined scenarios. The approach leverages new surveillance data points to generate an adaptive ensemble by dynamically rejecting the subset of individual models' trajectories that do not closely match the observed trends.

**Datasets**

We consider the projections submitted to the U.S. Flu Scenario Modeling Hub during Round 1 of 2023/2024 and 2024/2025 (https://fluscenariomodelinghub.org/index.html)

**Structure of the repo**

The folder structure of the repository is as follows:
- input_data: this folder contains all the data needed to reproduce the analysis.
- scenario_modeling: this folder contains the US national level and US states level analysis. Code, figures, and output data are included.
- short_term_forecasting: this folder contains code, figures, and output data as the result of the short-term forecasting retrospective analysis.

**System requirements**

- Operating system: Linux or macOS
- Python version: 3.9
- Python dependencies (see requirements.txt):
  - numpy==1.26.4
  - pandas==2.3.3
  - matplotlib==3.7.0
  - seaborn==0.13.2
  - scikit-learn==1.5.1
  - pyarrow==21.0.0
  - requests==2.32.3
  - rpy2==3.5.16  (requires R)
  - PyYAML==6.0.1
  - pillow==10.2.0
- R version: 4.3.2 (required for rpy2, used to compute the LOP ensembles in `ensemble_lop.r`)
- R dependencies:
  - hubEnsembles 0.1.8 and hubUtils 1.2.0.9000 (from GitHub: hubverse-org)
  - CombineDistributions 0.2.1 (from GitHub: eahowerton)
  - dplyr 1.1.4, purrr 1.1.0, jsonlite 2.0.0, yaml 2.3.10
- Hardware: no non-standard hardware required

**Installation Guide**

Clone or download the repository:

git clone https://github.com/sfiandrino/adaptive_ensemble.git
cd adaptive_ensemble

Create a Python virtual environment (optional but recommended):

python3 -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows

Install Python dependencies:
pip install -r requirements.txt

Install R (required: the ensembles are computed in R through rpy2):
Make sure R 4.3.2 is installed, then install the R packages from an R session:

install.packages(c("remotes", "dplyr", "purrr", "jsonlite", "yaml"))
remotes::install_github("hubverse-org/hubUtils")
remotes::install_github("hubverse-org/hubEnsembles")
remotes::install_github("eahowerton/CombineDistributions")

**Demo**

*Quick start: a minimal walk-through of the methodology*

If you want to see how the adaptive ensemble works before running the full analysis, start from scenario_modeling/US_national/code/demo_adaptive_ensemble.py. It is a reduced version of adaptive_ensemble2_S2.py that runs on the small demo dataset input_data/demo_trajectories.parquet (3 models, 6 scenarios, 100 trajectories each, 20 weekly horizons, season 2023/2024) and prints the steps of the methodology one by one:

- Step 0: the input pool of scenario trajectories.
- Step 1: the weekly loop, i.e. score every trajectory against the surveillance data available so far, keep the best top-k% of each model, and combine the retained trajectories with a Linear Opinion Pool. The first week is shown in full detail (best and worst trajectories, trajectories kept per model, posterior over scenarios, resulting ensemble quantiles), the remaining weeks are summarised one line each.
- Step 2: persistence of the selection across weeks (Jaccard similarity index).
- Step 3: posterior distribution over scenarios.

Run it from its own folder:

cd scenario_modeling/US_national/code
python demo_adaptive_ensemble.py

Notes:
- An internet connection is required, because the weekly (non-backfilled) surveillance snapshots are downloaded from the FluSight forecast hub.
- All results are written to demo_-prefixed folders (../output_data/demo_adaptive_ensemble2/, ../output_data/demo_persistence_analysis/, ../output_data/demo_posterior_analysis/).

**Instructions to run the full analysis**

All scripts and notebooks are run from their own `code/` folder (paths below are relative to the repository root). Input data are in `input_data/`; results are written to the `output_data/` and `figures/` folders of each analysis.

*Data downloaded at run time.* In addition to the files in `input_data/`, the code downloads from public GitHub repositories:
- the weekly (unrevised) surveillance snapshots of incident influenza hospital admissions, from the FluSight Forecast Hub target-data archive (https://github.com/cdcepi/FluSight-forecast-hub/tree/main/auxiliary-data/target-data-archive);
- the FluSight baseline and ensemble forecasts, from https://github.com/cdcepi/FluSight-forecast-hub (model-output, season 2024-2025) and from the archives https://github.com/cdcepi/Flusight-baseline and https://github.com/cdcepi/Flusight-ensemble (season 2023-2024).

An internet connection is therefore required to run the analyses.

*Workflow.* In each analysis, set the `season` variable ('2023-2024' or '2024-2025') at the top of the script or notebook and run the steps in this order:

1. US national scenario projections (`scenario_modeling/US_national/code/`)
   1. `original_ensembles_generation.py`: original (static) ensembles over the entire projection period (LOP, computed in R by `ensemble_lop.r`).
   2. `adaptive_ensemble2_S2.py`: adaptive ensemble, weekly trajectory selection (wMAPE or RMSE loss function).
   3. `adaptive_ensemble2_evaluation.ipynb`: WIS and MAE of the adaptive and original ensembles and their ratios.
   4. `adaptive_ensemble2_performance_visualization.ipynb`: performance figures.
   5. `scenario_posterior_visualization.ipynb`: posterior scenario analysis.
   6. `adaptive_ensemble2_S2_metricpersistance.py`, `persistance_visualization_JSI_acrossrounds.ipynb` and `persistance_JSI_acrossmetrics_wmape_rmse.ipynb`: persistence of the selected trajectories (Jaccard similarity index).
   7. `influenza_subtypes_plots_2024-25.ipynb`: observed influenza subtypes compared with the posterior scenario contributions.
   8. Sensitivity analyses: `bootstrap-analysis/` (`create_bootstrap_dataset.py`, `adaptive_ensemble2_S2_bootstrapping.py`, `adaptive_ensemble2_evaluation_bootstrap.ipynb`, `bootstrap_performance_visualization.ipynb`) and `age-strata-65plus-analysis/` (`original_ensembles_generation_agestrata_65+.py`, `adaptive_ensemble2_S2_agestrata_65+.py`, `adaptive_ensemble2_evaluation_agestrata_65+.ipynb`, `performance_visualization_agestrata_65+.ipynb`, `scenario_posterior_agestrata_65+.ipynb`, `influenza_subtypes_plots_2024-25_age65plus.ipynb`).
   9. `smh_models_spatialcoverage_seasons.ipynb` and `smh_models_age_stratification.py`: summary of the contributing SMH models (trajectories, geographic coverage, age stratification).
2. US states scenario projections (`scenario_modeling/US_states/code/`): `original_ensembles_generation.py`, `adaptive_ensemble2_S2.py`, `adaptive_ensemble2_evaluation.ipynb`, `adaptive_ensemble2_performance_visualization.ipynb`.
3. Short-term forecasting (`short_term_forecasting/code/`): `adaptive_ensemble2_S1_forecasting.py` (adaptive ensemble forecasts), `adaptive_ensemble2_forecasting_evaluation.ipynb` (WIS, MAE and coverage against the FluSight baseline and ensemble), `adaptive_ensemble2_performance_visualization.ipynb` (season 2024-2025) and `adaptive_ensemble2_performance_visualization_time.ipynb` (season 2023-2024), `adaptive_ensemble2_forecasts_visualization.ipynb` (forecast plots), `summary-flusight-models.ipynb` (summary of the FluSight models).

*Expected output:* data files (Parquet or CSV) in `output_data/` and the figures of the manuscript and Supplementary Information in `figures/`.

*Expected run time:* run time depends on the task; the state-level analyses take longer because each state is processed separately.

**Figures and tables: where they are produced**

| Manuscript item | Code (relative to the repository root) |
|---|---|
| Fig. 1 | Schematic drawn by the authors (no code) |
| Fig. 2, Table 1 | `scenario_modeling/US_national/code/adaptive_ensemble2_performance_visualization.ipynb` (figure); `scenario_modeling/US_national/code/adaptive_ensemble2_evaluation.ipynb` (table values); season 2024-2025 |
| Fig. 3 | `scenario_modeling/US_states/code/adaptive_ensemble2_performance_visualization.ipynb` (season 2024-2025, 25%) |
| Fig. 4 | `scenario_modeling/US_national/code/scenario_posterior_visualization.ipynb` (season 2024-2025) |
| Fig. 5, Table 2 | `short_term_forecasting/code/adaptive_ensemble2_performance_visualization.ipynb` (season 2024-2025) |
| Fig. 6 | Scenario definitions of the U.S. Flu Scenario Modeling Hub, Round 1 (table, no code) |
| Table 3 | `scenario_modeling/US_national/code/smh_models_age_stratification.py`, `smh_models_spatialcoverage_seasons.ipynb` and model metadata of the Flu Scenario Modeling Hub |

| Supplementary item | Code (relative to the repository root) |
|---|---|
| Table S1 | `scenario_modeling/US_national/code/smh_models_spatialcoverage_seasons.ipynb` |
| Table S2 | `short_term_forecasting/code/summary-flusight-models.ipynb` |
| Figs. S1-S7, Table S3 | `scenario_modeling/US_national/code/adaptive_ensemble2_performance_visualization.ipynb` (figures; S1, S3, S6: 2024-2025 or 2023-2024 as stated in the captions; S4-S5: RMSE loss function) and `adaptive_ensemble2_evaluation.ipynb` (Table S3); set `season` accordingly |
| Fig. S8 | `scenario_modeling/US_national/code/bootstrap-analysis/bootstrap_performance_visualization.ipynb` |
| Fig. S9 | `scenario_modeling/US_national/code/age-strata-65plus-analysis/performance_visualization_agestrata_65+.ipynb` |
| Figs. S10-S11 | `scenario_modeling/US_national/code/persistance_visualization_JSI_acrossrounds.ipynb` |
| Figs. S12-S16 | `scenario_modeling/US_states/code/adaptive_ensemble2_performance_visualization.ipynb` (set `season` and the percentage `k`) |
| Figs. S17-S21 | `scenario_modeling/US_national/code/influenza_subtypes_plots_2024-25.ipynb` (observed subtypes) and `scenario_posterior_visualization.ipynb` (posterior scenario contributions); set `season` |
| Fig. S22 | `scenario_modeling/US_national/code/age-strata-65plus-analysis/scenario_posterior_agestrata_65+.ipynb` and `influenza_subtypes_plots_2024-25_age65plus.ipynb` |
| Figs. S23-S24, S27-S28, Table S4 | `short_term_forecasting/code/adaptive_ensemble2_performance_visualization.ipynb` (figures) and `adaptive_ensemble2_forecasting_evaluation.ipynb` (Table S4); season 2024-2025 |
| Figs. S25-S26, S32-S33 | `short_term_forecasting/code/adaptive_ensemble2_forecasts_visualization.ipynb` (set `season`) |
| Figs. S29-S31, S34-S35, Tables S5-S6 | `short_term_forecasting/code/adaptive_ensemble2_performance_visualization_time.ipynb` and `adaptive_ensemble2_forecasting_evaluation.ipynb`; season 2023-2024 |

**Data licence and reuse**

All source code that is specific to the overall project is available under an open-source MIT license. This license does NOT cover model code from the various teams or model scenario data, which are available under specified licenses.
