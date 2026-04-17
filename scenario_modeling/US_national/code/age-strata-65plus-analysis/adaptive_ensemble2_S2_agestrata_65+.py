import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent / 'utils'))
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

def loading_surveillance_agestrata(ref_date, start_date, path_surv):
    """
    This function loads the surveillance data from a GitHub repository (non-backfilled data).
    input:
        ref_date = reference date for the data
        start_date = start date for the data
        github_repo = GitHub repository name
        github_directory = directory in the GitHub repository
    output:
        df_surv_date_US = DataFrame containing the surveillance data for the US
    """

    df_surv = pd.read_csv(path_surv)
    
    ref_date = pd.to_datetime(ref_date)
    start_date = pd.to_datetime(start_date)
    df_surv['date'] = pd.to_datetime(df_surv['date'])
    df_surv = df_surv[(df_surv['target'] == 'inc hosp') & (df_surv['age_group'] == '65-130')]
    # filter the dataframe to only include rows where the location is 'US' and the date is between start_date and ref_date
    df_surv_date_US = df_surv[(df_surv.location == 'US') & 
                            (df_surv.date <= ref_date) & 
                            (df_surv.date >= start_date)]
    df_surv_date_US = df_surv_date_US.rename(columns={"observation": "hospitalizations"})
    df_surv_date_US = df_surv_date_US.sort_values(by='date')
    df_surv_date_US['horizon'] = np.arange(1, len(df_surv_date_US)+1)
    return df_surv_date_US

def computing_wmape_trajs_agestrata(df, df_surv, list_hor):
    """
    This function computes the WMAPE for each trajectory in the dataframe.
    input:
        df = dataframe with the trajectories
        df_surv = dataframe with the surveillance data
        list_hor = list of horizons to consider
    output:
        dict_wmape_traj = dictionary with the WMAPE values for each trajectory
    """
    dict_wmape_traj = {}
    df = df.iloc[:-1]
    print(df)
    for col in df.columns:
        df_col = df[col]
        wmape_result = get_wmape(df_surv[df_surv.horizon.isin(list_hor)]['hospitalizations'].values, df_col.values)
        modelname = col.split('-2024-08-01')[0][:-2]
        dict_wmape_traj[col] = [modelname, wmape_result]
    return dict_wmape_traj
    
if __name__ == "__main__":
    path_to_save = "../../output_data/adaptive_ensemble2_agestrata/"
    path_R_script = "ensemble_lop_agestrata.r"
    path_surv = "../../../../input_data/target-data-age-stratification.csv"
    path_posterior = "../../output_data/posterior_analysis_agestrata/"
    df_scenarios = pd.read_parquet("../../../../input_data/SMH_projections_age65plus.parquet")
    df_scenarios['ids'] = df_scenarios['traj_id']
    print(df_scenarios.head())
    k_values = [0.05, 0.15, 0.25, 0.50, 0.75] # set the top k values for selecting trajectories to generate the adaptive ensemble 
    loss_function = 'wmape' # set the loss function to be used for ranking the trajectories
    is_original = False
    scenario = "Ens2"
    
    dict_keep_trajs = {}
    dict_posterior = {}
    df_scenarios['horizon'] = pd.to_numeric(df_scenarios['horizon'], errors='coerce').fillna(0).astype(int)
    df_scenarios = df_scenarios[df_scenarios['horizon'] >= 13].copy()
    horizon_map = {h: i+1 for i, h in enumerate(sorted(df_scenarios['horizon'].unique()))}
    df_scenarios['horizon'] = df_scenarios['horizon'].map(horizon_map)
    print(df_scenarios)
    #Define period of scenario projections that correspond to the round in the ensemble files
    start_date = datetime(2024, 11, 9)
    end_date = datetime(2025, 5, 10)
    date_list = [(start_date + timedelta(days=x)).strftime("%Y-%m-%d") for x in range(0, (end_date - start_date).days, 7)]
    end_round = len(date_list) #max value of horizon columns for df_scenarios
    df_scenarios = df_scenarios[df_scenarios.horizon <= end_round] 
    df_scenarios['target_end_date'] = df_scenarios['horizon'].apply(lambda x: date_list[x-1])
    #df_scenarios['output_type_id'] = df_scenarios['output_type_id'].astype(int)
    round_init = 2 # this is the first round due to surpassing epidemic threshold
    list_hor = list(range(1, round_init + 1)) #to include in the loss function computation also the first data point
    for h in df_scenarios['horizon'].unique().astype(int)[round_init:end_round]:
        list_hor.append(h)
        df_scenario_h = df_scenarios[df_scenarios['horizon'].isin(list_hor)]
        ref_date = pd.to_datetime(df_scenario_h.loc[df_scenario_h['horizon'] == h, 'target_end_date'].values[0]) # get reference date for current horizon
        ref_date_surveillance = (ref_date - timedelta(days=7)).date() #survaillance data (7 days before the reference date)
        
        df_surv = loading_surveillance_agestrata(ref_date_surveillance, start_date, path_surv)
        new_df = df_scenario_h.pivot(index='horizon', columns='ids', values='value')
        
        if loss_function == 'wmape':
            dict_score_traj = computing_wmape_trajs_agestrata(new_df, df_surv, list_hor)
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
    # POSTERIOR DISTRIBUTION ANALYSIS
    start_week = datetime.strptime("2024-11-09", "%Y-%m-%d")
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
    df_posteriors.to_csv(path_posterior + f"posterior_distribution_65plus_{loss_function}.csv", index=False)