import pandas as pd
import numpy as np
from functions import *
from rpy2.robjects import pandas2ri
from rpy2 import robjects as r
from datetime import timedelta, datetime
import warnings
import requests
from io import StringIO
from itertools import chain
# Suppress all warnings temporarily
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
if __name__ == "__main__":
    github_repo = "cdcepi/FluSight-forecast-hub"
    path_to_save = "../output_data/adaptive_ensemble2/"
    path_persistence = "../output_data/persistence_analysis/"
    path_posterior = "../output_data/posterior_analysis/"
    github_directory = "auxiliary-data/target-data-archive"
    path_R_script = "/home/sfiandrino/PhD_Project/adaptive_ensemble_methodological/scenario_modeling/US_national/code/ensemble_lop.r"
    df_scenarios = pd.read_csv("../../../input_data/SMH_trajectories_FluRound1_2023_2024.csv", index_col = 0)
    k_values = [0.05, 0.15, 0.25, 0.50, 0.75] # set the top k values for selecting trajectories to generate the adaptive ensemble 
    loss_function = 'rmse' # set the loss function to be used for ranking the trajectories
    is_original = False
    scenario = "Ens2"
    create_ID_ModelTrajectory(df_scenarios)
    dict_keep_trajs = {}
    dict_posterior = {}
    df_scenarios['horizon'] = pd.to_numeric(df_scenarios['horizon'], errors='coerce').fillna(0).astype(int)
    #Define period of scenario projections that correspond to the round in the ensemble files
    start_date = datetime(2023, 9, 9)
    end_date = datetime(2024, 4, 27)
    date_list = [(start_date + timedelta(days=x)).strftime("%Y-%m-%d") for x in range(0, (end_date - start_date).days, 7)]
    end_round = len(date_list) #max value of horizon columns for df_scenarios
    df_scenarios = df_scenarios[df_scenarios.horizon <= end_round] 
    df_scenarios['target_end_date'] = df_scenarios['horizon'].apply(lambda x: date_list[x-1])
    df_scenarios['output_type_id'] = df_scenarios['output_type_id'].astype(int)
    round_init = 8 # this is the first round due to surpassing epidemic threshold
    list_hor = list(range(1, round_init + 1)) #to include in the loss function computation also the first data point
    for h in df_scenarios['horizon'].unique().astype(int)[round_init:end_round]:
        list_hor.append(h)
        df_scenario_h = df_scenarios[df_scenarios['horizon'].isin(list_hor)]
        ref_date = pd.to_datetime(df_scenario_h.loc[df_scenario_h['horizon'] == h, 'target_end_date'].values[0]) # get reference date for current horizon
        ref_date_surveillance = (ref_date - timedelta(days=7)).date() #survaillance data (7 days before the reference date)
        df_surv = loading_surveillance(ref_date_surveillance, start_date, github_repo, github_directory)
        new_df = df_scenario_h.pivot(index='horizon', columns='ids', values='value')
        
        if loss_function == 'wmape':
            dict_score_traj = computing_wmape_trajs(new_df, df_surv, list_hor)
        else:
            dict_score_traj = computing_rmse(new_df, df_surv, list_hor)
        for k in k_values:
            perc_trajs_scenarios, df_toens, all_keeptrajs = ranking_trajs(dict_score_traj, k, df_scenarios)
            if k not in dict_keep_trajs.keys():
                dict_keep_trajs[k] = [all_keeptrajs]
                dict_posterior[k] = [perc_trajs_scenarios]
            else:
                dict_keep_trajs[k].append(all_keeptrajs)
                dict_posterior[k].append(perc_trajs_scenarios)
            models_names_k = df_toens.model_name.unique()
            dfQ_k = quantile_computation(df_toens, models_names_k)
            dfQ_k_r = pandas_to_r_dataframe(dfQ_k)
            pandas2ri.activate()
            r.r.source(path_R_script)
            dfQ_k_r_r = pandas2ri.py2rpy(dfQ_k_r)
            day_tosave = ref_date.strftime('%Y-%m-%d')
            ens_r = r.r['ensemble_lop'](dfQ_k_r_r, h, k, day_tosave, path_to_save, loss_function, is_original, scenario)
    
    # PERSISTENCE ANALYSIS
    jaccard_index_dict_to = compute_jaccard_indices(dict_keep_trajs, mode="to")
    jaccard_index_dict_prev_h = compute_jaccard_indices(dict_keep_trajs, mode="prev")
    df_jaccard_to = pd.DataFrame.from_dict(jaccard_index_dict_to)
    df_jaccard_prev_h = pd.DataFrame.from_dict(jaccard_index_dict_prev_h)
    df_jaccard_to.index = pd.date_range(start='2023-11-04', periods=len(df_jaccard_to), freq='W-SAT')
    df_jaccard_prev_h.index = pd.date_range(start='2023-11-11', periods=len(df_jaccard_prev_h), freq='W-SAT')
    df_jaccard_to.to_csv(path_persistence + f"Jaccard_index_t0_Ens2_S2_LOP_{loss_function}.csv")
    df_jaccard_prev_h.to_csv(path_persistence + f"Jaccard_index_tprevious_Ens2_S2_LOP_{loss_function}.csv")

    # POSTERIOR DISTRIBUTION ANALYSIS
    start_week = datetime.strptime("2023-11-04", "%Y-%m-%d")
    posterior_rows = []
    for k_value, week_list in dict_posterior.items():
        for week_idx, scenario_dict in enumerate(week_list):
            ref_date = start_week + timedelta(weeks=week_idx)
            for scenario, value in scenario_dict.items():
                posterior_rows.append({
                    "week": ref_date.strftime("%Y-%m-%d"),
                    "k": k_value,
                    "scenario": scenario,
                    "posterior_value": value
                })

    df_posteriors = pd.DataFrame(posterior_rows)
    df_posteriors.to_csv(path_posterior + f"posterior_distribution_Ens2_S2_LOP_{loss_function}.csv", index=False)