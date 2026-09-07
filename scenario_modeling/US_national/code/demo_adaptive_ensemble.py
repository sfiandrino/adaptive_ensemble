"""
demo_adaptive_ensemble.py
=========================

A minimal, self-contained walk-through of the *adaptive ensemble* methodology
(strategy S2, US national level), meant as an entry point for new users.

It is a stripped-down version of `adaptive_ensemble2_S2.py` that runs on the
small demo dataset `input_data/demo_trajectories.parquet` (3 models x 6
scenarios x 100 trajectories x 20 weekly horizons, season 2023-2024) and prints
only what is needed to follow the method.

The idea in one paragraph
-------------------------
Scenario projections are produced before a season starts, as ensembles of
stochastic trajectories, and are never revised. The adaptive ensemble revisits
them every week: as surveillance data accumulate, each individual trajectory is
scored against the observations already available (RMSE here), the worst ones
are discarded, and only the best top-k% per model are kept. The retained
trajectories are summarised into quantiles per model and combined with a
Linear Opinion Pool (LOP, run in R via `ensemble_lop.r`) into the ensemble
projection for the remaining weeks. Two diagnostics come for free: which
scenarios the retained trajectories come from (the *posterior* over scenarios)
and how stable the selection is from week to week (Jaccard similarity index).

Everything this script writes goes into `demo_`-prefixed folders:
    ../output_data/demo_adaptive_ensemble2/    weekly LOP ensemble quantiles
    ../output_data/demo_persistence_analysis/  Jaccard similarity indices
    ../output_data/demo_posterior_analysis/    posterior over scenarios

Requirements: pandas, numpy, scikit-learn, pyarrow, requests, rpy2, and R with
hubUtils / hubEnsembles / CombineDistributions. An internet connection is needed
to pull the weekly (non-backfilled) FluSight surveillance snapshots from GitHub.

Run with:   python demo_adaptive_ensemble.py
"""

import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from utils.functions import (
    compute_jaccard_indices,
    computing_rmse,
    create_ID_ModelTrajectory,
    loading_surveillance,
    pandas_to_r_dataframe,
    quantile_computation,
    ranking_trajs,
)

from rpy2 import robjects as r
from rpy2.robjects import pandas2ri

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration - everything you may want to change is here
# ---------------------------------------------------------------------------
DEMO_TRAJECTORIES = os.path.join(CODE_DIR, "../../../input_data/demo_trajectories.parquet")
PATH_R_SCRIPT = os.path.join(CODE_DIR, "ensemble_lop.r")

PATH_ENSEMBLE = os.path.join(CODE_DIR, "../output_data/demo_adaptive_ensemble2") + os.sep
PATH_PERSISTENCE = os.path.join(CODE_DIR, "../output_data/demo_persistence_analysis") + os.sep
PATH_POSTERIOR = os.path.join(CODE_DIR, "../output_data/demo_posterior_analysis") + os.sep

GITHUB_REPO = "cdcepi/FluSight-forecast-hub"
GITHUB_DIRECTORY = "auxiliary-data/target-data-archive"

SEASON = "2023-2024"          # the demo dataset only covers this season
SCENARIO = "Ens2"             # label of the scenario-round these trajectories come from
LOSS_FUNCTION = "rmse"        # 'rmse' or 'wmape' (see computing_wmape_trajs)
IS_ORIGINAL = False           # False -> adaptive ensemble, True -> original (never re-weighted) ensemble
K_VALUES = [0.05, 0.25, 0.50]  # top-k% of trajectories kept per model (paper also uses 0.15 and 0.75)
K_SHOWCASE = 0.25             # the single k detailed in the printout

# Weekly grid of the 2023-2024 projection round, and the first week at which the
# adaptive ensemble starts (epidemic activity above threshold).
START_DATE = datetime(2023, 9, 9)
END_DATE = datetime(2024, 4, 27)
ROUND_INIT = 8


def banner(text):
    print("\n" + "=" * 79)
    print(text)
    print("=" * 79)


def r_to_pandas(obj):
    """The LOP result comes back from R; make sure we hold a pandas DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj
    return pandas2ri.rpy2py(obj)


if __name__ == "__main__":

    for path in (PATH_ENSEMBLE, PATH_PERSISTENCE, PATH_POSTERIOR):
        os.makedirs(path, exist_ok=True)

    # -----------------------------------------------------------------------
    # STEP 0 - The input: an ensemble of scenario trajectories
    # -----------------------------------------------------------------------
    banner("STEP 0 | Input data: scenario projection trajectories")

    df_scenarios = pd.read_parquet(DEMO_TRAJECTORIES)
    df_scenarios = df_scenarios[df_scenarios["location"] == "US"]
    df_scenarios.rename(columns={"model_id": "model_name"}, inplace=True)
    df_scenarios["horizon"] = pd.to_numeric(df_scenarios["horizon"], errors="coerce").fillna(0).astype(int)

    # Each single trajectory gets a unique id: scenario_id + sample_id + model_name.
    # This id is the unit the adaptive ensemble selects on.
    create_ID_ModelTrajectory(df_scenarios, SEASON)
    df_scenarios["output_type_id"] = df_scenarios["output_type_id"].astype(int)

    # Weekly calendar of the projection round; horizon h -> target_end_date.
    date_list = [(START_DATE + timedelta(days=x)).strftime("%Y-%m-%d")
                 for x in range(0, (END_DATE - START_DATE).days, 7)]
    end_round = len(date_list)
    df_scenarios = df_scenarios[df_scenarios.horizon <= end_round]
    df_scenarios["target_end_date"] = df_scenarios["horizon"].apply(lambda x: date_list[x - 1])

    # Lookup used only for the printout: `computing_rmse` derives the model label by
    # splitting the id on "_", which truncates names that contain underscores.
    id_to_model = df_scenarios.drop_duplicates("ids").set_index("ids")["model_name"]

    models = sorted(df_scenarios.model_name.unique())
    scenarios = sorted(df_scenarios.scenario_id.unique())
    horizons = sorted(df_scenarios.horizon.unique())
    n_traj_per_model = df_scenarios.groupby("model_name")["ids"].nunique()

    print(f"file            : {os.path.relpath(DEMO_TRAJECTORIES, CODE_DIR)}")
    print(f"season          : {SEASON}   target: {df_scenarios.target.unique()[0]}   location: US")
    print(f"models          : {len(models)}  -> {', '.join(models)}")
    print(f"scenarios       : {len(scenarios)}  -> {', '.join(scenarios)}")
    print(f"horizons        : {len(horizons)} weeks, {date_list[horizons[0] - 1]} -> {date_list[horizons[-1] - 1]}")
    print(f"trajectories    : {df_scenarios.ids.nunique()} in total "
          f"({', '.join(f'{m}: {n}' for m, n in n_traj_per_model.items())})")
    print("\nA trajectory id is 'scenario_id + sample_id + model_name'; one row per (id, horizon):")
    print(df_scenarios[["ids", "model_name", "scenario_id", "horizon", "target_end_date", "value"]].head(3)
          .to_string(index=False))

    # -----------------------------------------------------------------------
    # STEP 1 - Weekly loop: score -> rank -> select -> combine
    # -----------------------------------------------------------------------
    banner("STEP 1 | Weekly adaptive selection and Linear Opinion Pool")
    print(f"loss function   : {LOSS_FUNCTION}")
    print(f"top-k% kept     : {K_VALUES}   (per model, over all its trajectories)")
    print(f"first round     : week {ROUND_INIT + 1} ({date_list[ROUND_INIT]}), i.e. once surveillance data "
          f"are informative")
    print("\nAt each round the ensemble sees observations up to the previous week and")
    print("projects the remaining weeks of the season; the trajectory pool is re-selected")
    print("from scratch every time.")

    pandas2ri.activate()
    # defines ensemble_lop(); sourced once for the whole run
    r.r(f'suppressWarnings(suppressPackageStartupMessages(source("{PATH_R_SCRIPT}")))')

    dict_keep_trajs = {}   # k -> [list of retained trajectory ids, one entry per round]
    dict_posterior = {}    # k -> [{scenario: share of retained trajectories}, one per round]
    round_summary = []
    list_hor = list(range(1, ROUND_INIT + 1))
    first_round = True

    for h in df_scenarios["horizon"].unique().astype(int)[ROUND_INIT:end_round]:
        list_hor.append(h)
        df_scenario_h = df_scenarios[df_scenarios["horizon"].isin(list_hor)]
        ref_date = pd.to_datetime(df_scenario_h.loc[df_scenario_h["horizon"] == h, "target_end_date"].values[0])

        # Surveillance data as they were known one week before the reference date
        # (non-backfilled snapshot, exactly what a real-time user would have had).
        ref_date_surveillance = (ref_date - timedelta(days=7)).date()
        df_surv = loading_surveillance(ref_date_surveillance, START_DATE, GITHUB_REPO, GITHUB_DIRECTORY, SEASON)

        # One column per trajectory, one row per horizon -> score every trajectory
        # against the observations available so far.
        new_df = df_scenario_h.pivot(index="horizon", columns="ids", values="value")
        dict_score_traj = computing_rmse(new_df, df_surv, list_hor, SEASON)

        if first_round:
            print("\n" + "-" * 79)
            print(f"ROUND 1 in detail | reference date {ref_date.date()} (horizon h={h})")
            print("-" * 79)
            print(f"observations used for scoring : {len(df_surv[df_surv.horizon.isin(list_hor)])} weeks "
                  f"({df_surv.date.min().date()} -> {df_surv.date.max().date()})")
            print(f"trajectories scored           : {len(dict_score_traj)}")
            df_sc = pd.DataFrame.from_dict(dict_score_traj, orient="index",
                                           columns=["model_name", LOSS_FUNCTION]).sort_values(LOSS_FUNCTION)
            df_sc["model_name"] = id_to_model.reindex(df_sc.index).values
            print(f"\nbest 3 trajectories by {LOSS_FUNCTION}:")
            print(df_sc.head(3).to_string())
            print(f"worst 3 trajectories by {LOSS_FUNCTION}:")
            print(df_sc.tail(3).to_string())

        for k in K_VALUES:
            # Keep the best k% of trajectories *within each model*, so that no model
            # can be silenced entirely by the selection.
            perc_trajs_scenarios, df_toens, all_keeptrajs = ranking_trajs(dict_score_traj, k, df_scenarios, SEASON)
            dict_keep_trajs.setdefault(k, []).append(all_keeptrajs)
            dict_posterior.setdefault(k, []).append(perc_trajs_scenarios)

            # Summarise the retained trajectories of each model into quantiles,
            # then combine the models with a Linear Opinion Pool (in R).
            models_names_k = df_toens.model_name.unique()
            dfQ_k = quantile_computation(df_toens, models_names_k)
            dfQ_k_r = pandas2ri.py2rpy(pandas_to_r_dataframe(dfQ_k))
            day_tosave = ref_date.strftime("%Y-%m-%d")
            ens_r = r.r["ensemble_lop"](dfQ_k_r, h, k, day_tosave, PATH_ENSEMBLE,
                                        LOSS_FUNCTION, IS_ORIGINAL, SCENARIO, SEASON)
            df_ens = r_to_pandas(ens_r)
            df_ens.to_csv(f"{PATH_ENSEMBLE}{day_tosave}_{k}_{LOSS_FUNCTION}_{SEASON}.csv", index=False)

            if first_round and k == K_SHOWCASE:
                kept = id_to_model.reindex(all_keeptrajs).value_counts()
                print(f"\nselection with k={k:.0%} -> {len(all_keeptrajs)} trajectories kept out of "
                      f"{len(dict_score_traj)}")
                print("  per model  : " + ", ".join(f"{m}: {n}" for m, n in kept.sort_index().items()))
                print("  posterior over scenarios (share of retained trajectories):")
                for sc, val in perc_trajs_scenarios.items():
                    print(f"      {sc}  {val:6.1%}  {'#' * int(round(val * 50))}")
                qcol = next(c for c in df_ens.columns if c.lower().startswith("quantile"))
                piv = df_ens.pivot(index="horizon", columns=qcol, values="value")
                sel = [c for c in (0.025, 0.5, 0.975) if c in piv.columns]
                print("\n  resulting LOP ensemble (median and 95% projection interval), first weeks:")
                print(piv[sel].head(4).round(0).to_string())

            if k == K_SHOWCASE:
                top_scenario = max(perc_trajs_scenarios, key=perc_trajs_scenarios.get)
                qcol = next(c for c in df_ens.columns if c.lower().startswith("quantile"))
                median_h = df_ens[(df_ens.horizon == h) & (np.isclose(df_ens[qcol], 0.5))]["value"]
                round_summary.append({
                    "reference_date": ref_date.date(),
                    "h": h,
                    "obs_weeks": len(df_surv[df_surv.horizon.isin(list_hor)]),
                    "kept": len(all_keeptrajs),
                    "top_scenario": f"{top_scenario[0]} ({perc_trajs_scenarios[top_scenario]:.0%})",
                    "ens_median_h": round(float(median_h.iloc[0])) if len(median_h) else np.nan,
                })

        first_round = False

    print("\n" + "-" * 79)
    print(f"ALL ROUNDS | one line per week, shown for k={K_SHOWCASE:.0%}")
    print("-" * 79)
    df_rounds = pd.DataFrame(round_summary)
    print(df_rounds.to_string(index=False))
    print("\n'top_scenario' is the scenario contributing most retained trajectories that week:")
    print("it moves as data accumulate, which is precisely the adaptive part of the method.")

    # -----------------------------------------------------------------------
    # STEP 2 - Persistence: how stable is the set of retained trajectories?
    # -----------------------------------------------------------------------
    banner("STEP 2 | Persistence of the selection (Jaccard similarity index)")

    jaccard_index_dict_to = compute_jaccard_indices(dict_keep_trajs, mode="to")
    jaccard_index_dict_prev_h = compute_jaccard_indices(dict_keep_trajs, mode="prev")
    df_jaccard_to = pd.DataFrame.from_dict(jaccard_index_dict_to)
    df_jaccard_prev_h = pd.DataFrame.from_dict(jaccard_index_dict_prev_h)
    df_jaccard_to.index = pd.date_range(start="2023-11-04", periods=len(df_jaccard_to), freq="W-SAT")
    df_jaccard_prev_h.index = pd.date_range(start="2023-11-11", periods=len(df_jaccard_prev_h), freq="W-SAT")

    print("JSI vs the FIRST round (columns = k): 1 = same trajectories as at the start, 0 = fully renewed")
    print(df_jaccard_to.round(2).to_string())
    print("\nJSI vs the PREVIOUS round (mean over the season):")
    print(df_jaccard_prev_h.mean().round(3).to_string())

    df_jaccard_to.to_csv(PATH_PERSISTENCE + f"Jaccard_index_t0_{SCENARIO}_S2_LOP_{LOSS_FUNCTION}_{SEASON}.csv")
    df_jaccard_prev_h.to_csv(PATH_PERSISTENCE + f"Jaccard_index_tprevious_{SCENARIO}_S2_LOP_{LOSS_FUNCTION}_{SEASON}.csv")

    # -----------------------------------------------------------------------
    # STEP 3 - Posterior distribution over scenarios
    # -----------------------------------------------------------------------
    banner("STEP 3 | Posterior distribution over scenarios")

    start_week = datetime.strptime("2023-11-04", "%Y-%m-%d")
    posterior_rows = []
    for k_value, week_list in dict_posterior.items():
        for week_idx, scenario_dict in enumerate(week_list):
            ref_date = start_week + timedelta(weeks=week_idx)
            for scenario_name, value in scenario_dict.items():
                posterior_rows.append({
                    "week": ref_date.strftime("%Y-%m-%d"),
                    "k": k_value,
                    "scenario": scenario_name,
                    "posterior_value": value,
                })
    df_posteriors = pd.DataFrame(posterior_rows)
    df_posteriors.to_csv(
        PATH_POSTERIOR + f"posterior_distribution_{SCENARIO}_S2_LOP_{LOSS_FUNCTION}_{SEASON}.csv", index=False)

    print(f"Share of retained trajectories per scenario, week by week (k={K_SHOWCASE:.0%}):")
    print(df_posteriors[df_posteriors.k == K_SHOWCASE]
          .pivot(index="week", columns="scenario", values="posterior_value").round(2).to_string())

    # -----------------------------------------------------------------------
    banner("Output files")
    for path in (PATH_ENSEMBLE, PATH_PERSISTENCE, PATH_POSTERIOR):
        files = sorted(os.listdir(path))
        print(f"{os.path.relpath(path, CODE_DIR)}  ->  {len(files)} file(s)")
        for f in files[:3]:
            print(f"    {f}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more")
    print("\nFor the full analysis (all k values, both seasons, wmape/rmse), see adaptive_ensemble2_S2.py.")
