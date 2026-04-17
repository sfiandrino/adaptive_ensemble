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
    season = "2024-2025"
    github_repo = "cdcepi/FluSight-forecast-hub"
    path_to_save = "../output_data/adaptive_ensemble2/"
    github_directory = "auxiliary-data/target-data-archive"
    path_persistence = "../output_data/persistence_analysis/"
    if season == '2023-2024':
        df_scenarios = pd.read_csv("../../input_data/SMH_trajectories_FluRound1_2023_2024.csv", index_col = 0)
        start_date = datetime(2023, 9, 9)
        end_date = datetime(2024, 4, 27)
        round_init = 5
    elif season == '2024-2025':
        df_scenarios = pd.read_parquet("../../input_data/SMH_trajectories_FluRound1_2024_2025.parquet")
        start_date = datetime(2024, 8, 17) # the first round of the ensemble starts on this date
        end_date = datetime(2025, 6, 7) # the last round of the ensemble ends on this date 
        round_init = 14 #season 24/25 Flusight season starts from round 14 with data from round 13
        df_scenarios = df_scenarios[df_scenarios.location == 'US']
    df_scenarios.rename(columns={'model_id': 'model_name'}, inplace=True)
    k_values = [0.05, 0.15, 0.25, 0.50, 0.75] # set the top k values for selecting trajectories to generate the adaptive ensemble 
    create_ID_ModelTrajectory(df_scenarios, season)
    dict_score_allh = {}
    dict_keep_trajs = {}
    df_scenarios['horizon'] = pd.to_numeric(df_scenarios['horizon'], errors='coerce').fillna(0).astype(int)
    date_list = [start_date + timedelta(days=x) for x in range(0, (end_date-start_date).days, 7)]
    date_list = [date.strftime("%Y-%m-%d") for date in date_list]
    end_round = len(date_list)
    df_scenarios = df_scenarios[df_scenarios.horizon <= end_round] 
    df_scenarios['target_end_date'] = df_scenarios['horizon'].apply(lambda x: date_list[x-1])
    if season != "2024-2025":
        df_scenarios['output_type_id'] = df_scenarios['output_type_id'].astype(int)
    #round_init = 5 #season 23/24 Flusight season starts from round 4 with data from round 3
    #round_init = 14 #season 24/25 Flusight season starts from round 14 with data from round 13
    list_hor = list(range(1, round_init + 1)) # to include in the loss function computation also the first data point
    for h in df_scenarios['horizon'].unique().astype(int)[round_init:end_round]:
        list_hor.append(h)
        df_scenario_h = df_scenarios[df_scenarios['horizon'].isin(list_hor)]
        ref_date = pd.to_datetime(df_scenario_h.loc[df_scenario_h['horizon'] == h, 'target_end_date'].values[0]) # get reference date for current horizon
        print(f"Processing horizon {h} with reference date {ref_date}")
        if ref_date == (datetime(2025, 1, 25)):
            continue
        ref_date_surveillance = (ref_date - timedelta(days=7)).date() #survaillance data (7 days before the reference date)
        df_surv = loading_surveillance(ref_date_surveillance, start_date, github_repo, github_directory, season)
        new_df = df_scenario_h.pivot(index='horizon', columns='ids', values='value')
        dict_score_traj = computing_wmape_trajs(new_df, df_surv, list_hor, season)
        dict_score_allh[h] = dict_score_traj
        for k in k_values:
            perc_trajs_scenarios, df_toens, all_keeptrajs = ranking_trajs_forecastingS1(dict_score_traj, k, df_scenarios, season)
            if k not in dict_keep_trajs.keys():
                dict_keep_trajs[k] = [all_keeptrajs]
            else:
                dict_keep_trajs[k].append(all_keeptrajs)
            df_Q = quantile_computation(df_toens)
            df_Q = df_Q[(df_Q.horizon >= h) & (df_Q.horizon <= h+3)]
            df_Q = df_Q.to_csv(f"../output_data/adaptive_ensemble2_forecasts/{ref_date.strftime('%Y-%m-%d')}_{k}_{season}.csv")