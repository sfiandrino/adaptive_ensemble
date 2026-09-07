**Adaptive Ensemble**

**Introduction**

The adaptive ensemble approach combines epidemic projections from the Flu Scenario Modeling Hub across all defined scenarios. The approach leverages new surveillance data points to generate an adaptive ensemble by dynamically rejecting the subset of individual models' trajectories that do not closely match the observed trends.

**Datasets**

We consider the projections submitted to the U.S. Flu Scenario Modeling Hub during Round 1 of 2023/2024 (https://fluscenariomodelinghub.org/index.html)

**Structure of the repo**

The folder structure of the repository is as follows:
- input_data: this folder contains all the data needed to reproduce the analysis.
- scenario_modeling: this folder contains the US national level and US states level analysis. Code, figures, and output data are included.
- short_term_forecasting: this folder contains code, figures, and output data as the result of the short-term forecasting retrospective analysis.

**System requirements**

- Operating system: Linux
- Python version: 3.9.7
- Python dependencies: see requirements.txt for full list
  - matplotlib==3.10.8
  - numpy==2.4.1
  - pandas==2.3.3
  - pillow==12.1.0
  - requests==2.32.5
  - rpy2==3.6.4  (requires R ≥ 3.0)
  - seaborn==0.13.2
- R version: ≥ 3.0 (required for rpy2)
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

Install R (optional, required only for rpy2):
Make sure R ≥ 3.0 is installed.

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

Instructions to run on data
- Ensure that all dependencies are installed (pip install -r requirements.txt).
- Use the provided dataset in the data/ folder
- Run the main analysis script:
  - original_ensembles_generation.py: This code runs the original ensemble over the entire period. 
  - adaptive_ensemble2_S2.py: This code runs the adaptive ensemble process over the entire period.
  - adaptive_ensemble2_evaluation.ipynb: This code runs the evaluation analysis comparing the adaptive ensemble performance and the original ensemble performance.
  - adaptive_ensemble2_performance_visualization.ipynb: This code runs the analysis to reproduce the figures of the results.
  - scenario_posterior_visualization.ipynb: This code runs the analysis to reproduce the figures of the results.
  - persistance_visualization.ipynb: This code runs the analysis to reproduce the figures of the results.
  - adaptive_ensemble2_S1_forecasting.py: This code runs the short-term forecasting application. Equivalent code for evaluation and visualizations is provided for this specific application.
- Expected output:
  - Data files (Parquet or CSV)
  - Figures reproducing the ones shown in the manuscript
- Expected run time:
  - Run time may vary depending on the task and data analyzed. Subnational-level analyses require more time due to the multi-state processing.

**Data licence and reuse**

All source code that is specific to the overall project is available under an open-source MIT license. This license does NOT cover model code from the various teams or model scenario data, which are available under specified licenses.