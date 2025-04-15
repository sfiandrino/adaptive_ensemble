import pandas as pd
import numpy as np
from rpy2.robjects import pandas2ri
from rpy2 import robjects as r
from datetime import timedelta, datetime
from functions import *
import warnings
import requests
from io import StringIO
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    github_repo = "cdcepi/FluSight-forecast-hub"
    path_to_save = "../output_data/adaptive_ensemble2/"
    github_directory = "auxiliary-data/target-data-archive"
    path_persistence = "../output_data/persistence_analysis/"
    df_scenarios = pd.read_csv("../../input_data/SMH_trajectories_FluRound1_2023_2024.csv", index_col = 0)
    k_values = [0.05, 0.15, 0.25, 0.50, 0.75] # set the top k values for selecting trajectories to generate the adaptive ensemble 
    create_ID_ModelTrajectory(df_scenarios)
    dict_score_allh = {}
    dict_keep_trajs = {}
    df_scenarios['horizon'] = pd.to_numeric(df_scenarios['horizon'], errors='coerce').fillna(0).astype(int)
    #Define period of scenario projections that correspond to the round in the ensemble files
    start_date = datetime(2023, 9, 9)
    end_date = datetime(2024, 5, 11)
    date_list = [start_date + timedelta(days=x) for x in range(0, (end_date-start_date).days, 7)]
    date_list = [date.strftime("%Y-%m-%d") for date in date_list]
    end_round = len(date_list)
    df_scenarios = df_scenarios[df_scenarios.horizon <= end_round] 
    df_scenarios['target_end_date'] = df_scenarios['horizon'].apply(lambda x: date_list[x-1])
    df_scenarios['output_type_id'] = df_scenarios['output_type_id'].astype(int)
    round_init = 5 #Flusight season starts from round 4 with data from round 3
    list_hor = [1, 2, 3, 4, 5] 
    for h in df_scenarios['horizon'].unique().astype(int)[round_init:end_round]:
        list_hor.append(h)
        df_scenario_h = df_scenarios[df_scenarios['horizon'].isin(list_hor)]
        ref_date = pd.to_datetime(df_scenario_h.loc[df_scenario_h['horizon'] == h, 'target_end_date'].values[0]) # get reference date for current horizon
        ref_date_surveillance = (ref_date - timedelta(days=7)).date() #survaillance data (7 days before the reference date)
        df_surv = loading_surveillance(ref_date_surveillance, start_date, github_repo, github_directory)
        new_df = df_scenario_h.pivot(index='horizon', columns='ids', values='value')
        dict_score_traj = computing_wmape_trajs(new_df, df_surv, list_hor)
        dict_score_allh[h] = dict_score_traj
        for k in k_values:
            perc_trajs_scenarios, df_toens, all_keeptrajs = ranking_trajs_forecastingS1(dict_score_traj, k, df_scenarios)
            if k not in dict_keep_trajs.keys():
                dict_keep_trajs[k] = [all_keeptrajs]
            else:
                dict_keep_trajs[k].append(all_keeptrajs)
            df_Q = quantile_computation(df_toens)
            df_Q = df_Q[(df_Q.horizon >= h) & (df_Q.horizon <= h+3)]
            df_Q = df_Q.to_csv(f"../output_data/adaptive_ensemble2_forecasts/{ref_date.strftime('%Y-%m-%d')}_{k}.csv")
    # PERSISTENCE ANALYSIS
    jaccard_index_dict_to = compute_jaccard_indices(dict_keep_trajs, mode="to")
    jaccard_index_dict_prev_h = compute_jaccard_indices(dict_keep_trajs, mode="prev")
    df_jaccard_to = pd.DataFrame.from_dict(jaccard_index_dict_to)
    df_jaccard_prev_h = pd.DataFrame.from_dict(jaccard_index_dict_prev_h)
    df_jaccard_to.index = pd.date_range(start='2023-10-14', periods=len(df_jaccard_to), freq='W-SAT')
    df_jaccard_prev_h.index = pd.date_range(start='2023-10-21', periods=len(df_jaccard_prev_h), freq='W-SAT')
    df_jaccard_to.to_csv(path_persistence + "Jaccard_index_t0_Ens2_S1_forecasting.csv")
    df_jaccard_prev_h.to_csv(path_persistence + "Jaccard_index_tprevious_Ens2_S1_forecasting.csv")

    