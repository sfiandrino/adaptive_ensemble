import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rpy2.robjects import pandas2ri
from rpy2 import robjects as r
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import json
import requests
from io import StringIO

def define_parameters():
    github_repo = "cdcepi/FluSight-forecast-hub" # for influenza
    github_directory = "auxiliary-data/target-data-archive"
    season = '2024-2025' # or 2023-2024 for influenza
    loss_function = 'wmape'
    k_values = [0.05, 0.15, 0.25, 0.50, 0.75]  # k values for adaptive ensemble
    analysis_type = 'US_national' #or US_states for the state level analysis
    path_R_script_original = "ensemble_lop_bootstrap_original.r"
    path_R_script = "ensemble_lop_bootstrap.r"
    # this is the case of flu 2024-2025 
    scenarios_ids = ['A-2024-08-01', 'B-2024-08-01', 'C-2024-08-01', 'D-2024-08-01', 'E-2024-08-01', 'F-2024-08-01']
    num_tot_bootstrap = 10
    return github_repo, github_directory, season, loss_function, k_values, num_tot_bootstrap, analysis_type, path_R_script_original, path_R_script, scenarios_ids

def load_dataframe_smh(season):
    df_scenarios = pd.read_parquet("../../../../input_data/SMH_trajectories_FluRound1_2024_2025.parquet")
    df_scenarios.rename(columns={'model_id': 'model_name'}, inplace=True)
    return df_scenarios

def get_paths_tosave(analysis_type, df_scenarios, season):
    if analysis_type == 'US_national':
        states_to_process = ['US']
        path_list_trajs = f"../../output_data/list_trajs_bootstrap/"
        path_adaptive = f"../../output_data/adaptive_ensemble_bootstrap/"
        path_original = f"../../output_data/original_ensembles_bootstrap/"
        path_individualmodels = f"../../output_data/individual_models_quantiles_bootstrap/"
    return states_to_process, path_list_trajs, path_adaptive, path_original, path_individualmodels

# ======================== TRAJECTORIES IDs ========================
def create_ID_ModelTrajectory(df, season):
    """
    This function creates a unique identifier for each trajectory in the dataframe.
    The identifier is a combination of the scenario_id, output_type_id and model_name.
    input:
        df = dataframe with all trajectories
    output:
        df = dataframe with a new column that identifies each trajectory
    """
    df['ids'] = df['traj_id']
    return df

def read_csv_from_github(url):
    """
    Reads a CSV file from a given GitHub URL and returns it as a pandas DataFrame.
    input:
        url = URL of the CSV file on GitHub
    output:
        df = DataFrame containing the CSV data
    """
    response = requests.get(url)
    if response.status_code == 200:
        csv_content = response.content.decode('utf-8')
        return pd.read_csv(StringIO(csv_content))
    else:
        print(f"Failed to fetch file. Status code: {response.status_code}, Message: {response.text}")
        return pd.DataFrame()

def loading_surveillance(ref_date, start_date, github_repo, github_directory,state):
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
    file = f"target-hospital-admissions_{ref_date}.csv"
    file_url = f"https://raw.githubusercontent.com/{github_repo}/main/{github_directory}/{file}"
    df_surv = read_csv_from_github(file_url)
    ref_date = pd.to_datetime(ref_date)
    start_date = pd.to_datetime(start_date)
    df_surv['date'] = pd.to_datetime(df_surv['date'])
    # filter the dataframe to only include rows where the location is 'US' and the date is between start_date and ref_date
    df_surv_date_state = df_surv[(df_surv.location == state) & 
                            (df_surv.date <= ref_date) & 
                            (df_surv.date >= start_date)]
    df_surv_date_state = df_surv_date_state.rename(columns={"value": "hospitalizations"})
    df_surv_date_state = df_surv_date_state.sort_values(by='date')
    df_surv_date_state['horizon'] = np.arange(1, len(df_surv_date_state)+1)
    return df_surv_date_state

def load_surveillance_data(season, df_state, github_repo, github_directory, state):
    df_state['horizon'] = pd.to_numeric(df_state['horizon'], errors='coerce').fillna(0).astype(int)
    start_date = datetime(2024, 8, 17) # the first round of the ensemble starts on this date
    end_date = datetime(2025, 6, 7) # the last round of the ensemble ends on this date 
    date_list = [start_date + timedelta(days=x) for x in range(0, (end_date-start_date).days, 7)]
    date_list = [date.strftime("%Y-%m-%d") for date in date_list]
    df_state = df_state[df_state.horizon <= len(date_list)]
    df_state['target_end_date'] = df_state['horizon'].apply(lambda x: date_list[x-1])
    ref_date = pd.to_datetime(df_state['target_end_date'].max()) # get the last date in the scenarios dataframe
    ref_date_surveillance = (ref_date).date()
    df_surv = loading_surveillance(ref_date_surveillance, start_date, github_repo, github_directory, state)
    return df_state, df_surv, end_date

# ======================== QUANTILE COMPUTATION ========================
def quantile_computation(df, models_names):
    """
    This function computes quantiles for each model and horizon in the dataframe.
    input:
        df = dataframe with all trajectories
        models_names = list of model names
    output:
        dfQ = dataframe with quantiles for each model and horizon
    """
    quantiles = np.concatenate(([0.01, 0.025], np.round(np.arange(0.05, 1, 0.05), 3), [0.975, 0.99]))
    dfQ = pd.DataFrame()
    for h in df.horizon.unique():
        df_h = df[df.loc[:, 'horizon'] == h]
        for model in models_names:
            df_quantiles = pd.DataFrame()
            df_mod = df_h[df_h.loc[:, 'model_name'] == model]
            value_distribution = sorted(df_mod['value'].values)
            quantiles_values = np.quantile(value_distribution, quantiles)
            df_quantiles['quantiles'] = quantiles
            df_quantiles['value'] = quantiles_values
            df_quantiles['model_name'] = [model] * quantiles_values.shape[0]
            df_quantiles['horizon'] = [h] * quantiles_values.shape[0]
            dfQ = pd.concat([dfQ, df_quantiles])
    return dfQ

# ======================== R CALL WRAPPERS ========================
def pandas_to_r_dataframe(df):
    """
    This function converts a pandas dataframe to an R dataframe.
    input:
        df = dataframe to be converted
    output:
        df = dataframe converted to R dataframe
    """
    return pandas2ri.PandasDataFrame(df)
# ======================== QUANTILE COMPUTATION ========================
def quantile_computation_original(df):
    quantiles = np.concatenate(([0.01, 0.025], np.arange(0.05, 1, 0.05), [0.975, 0.99]))
    dfQ = pd.DataFrame()
    for (h, model), group in df.groupby(['horizon', 'model_name']):
        quantiles_values = np.quantile(group['value'], quantiles)
        dfQ = pd.concat([dfQ, pd.DataFrame({
            'quantiles': quantiles,
            'value': quantiles_values,
            'model_name': model,
            'horizon': h
        })])
    return dfQ.reset_index(drop=True)

def computing_ensemble(dfQ_r, scenario, path_R_script_original, day_to_save, loss_function, is_original, hor, top_traj, season, state):
    """
    Esegue l'ensemble LOP tramite uno script R.
    """
    pandas2ri.activate()

    # Legge ed esegue lo script R, esportando le funzioni
    with open(path_R_script_original, "r") as f:
        r_code = f.read()
    r_funcs = SignatureTranslatedAnonymousPackage(r_code, "r_funcs")

    dfQ_r_r = pandas2ri.py2rpy(dfQ_r)
    if scenario != "Ens2":
        scenario = scenario[0]

    ens_r = r_funcs.ensemble_lop(
        dfQ_r_r, hor, top_traj, day_to_save, loss_function, is_original,
        scenario, season, state
    )
    return ens_r


# ======================== ORIGINAL ENSEMBLE GENERATION ========================
def generate_original_ensembles(df, season, state, path_R_script_original):
    df_Q_all_scenarios, df_Q_all_ens2 = pd.DataFrame(), pd.DataFrame()
    for scenario in df['scenario_id'].unique():
        df_scenario = df[df['scenario_id'] == scenario]
        dfQ = quantile_computation_original(df_scenario)
        ens_r = computing_ensemble(pandas_to_r_dataframe(dfQ), scenario, path_R_script_original, "", "", True, "", " ", season, state)
        df_r = pandas2ri.rpy2py(ens_r)
        df_r['scenario_id'] = scenario
        df_Q_all_scenarios = pd.concat([df_Q_all_scenarios, df_r])

    dfQ_ens2 = quantile_computation_original(df)
    ens_r_ens2 = computing_ensemble(pandas_to_r_dataframe(dfQ_ens2), "Ens2", path_R_script_original, "", "", True, "", " ", season, state)
    df_Q_all_ens2 = pandas2ri.rpy2py(ens_r_ens2)
    df_Q_all_ens2['scenario_id'] = "Ens2"
    return df_Q_all_scenarios, df_Q_all_ens2


# ========================  INDIVIDUAL MODELS2 QUANTILES GENERATION ========================
def individual_model_quantile_computation(df_bootstrapped_state):   
    dfQ_individualmodel_allmodels = pd.DataFrame()
    for model in df_bootstrapped_state['model_name'].unique():
        df_scenarios_model_state = df_bootstrapped_state[df_bootstrapped_state['model_name'] == model]
        # Generate original ensemble for single scenarios
        dfQ_individualmodel_concat = pd.DataFrame()
        for scenario in df_scenarios_model_state.scenario_id.unique():
            df_scenario = df_scenarios_model_state[df_scenarios_model_state.loc[:, 'scenario_id'] == scenario]
            dfQ_individualmodel = quantile_computation(df_scenario)
            dfQ_individualmodel['model_name'] = model
            dfQ_individualmodel['scenario_id'] = scenario
            dfQ_individualmodel_concat = pd.concat([dfQ_individualmodel_concat, dfQ_individualmodel], ignore_index=True)

            # Generate original ensemble for the Ensemble2
        dfQ_individualmodel_ens2 = quantile_computation(df_scenarios_model_state)
        dfQ_individualmodel_ens2['model_name'] = model
        dfQ_individualmodel_ens2['scenario_id'] = 'Ens2'
        dfQ_individualmodel_concat = pd.concat([dfQ_individualmodel_concat, dfQ_individualmodel_ens2], ignore_index=True)
        # concat df_individualmodel and dfQ_individualmodel_ens2
        dfQ_individualmodel_allmodels = pd.concat([dfQ_individualmodel_allmodels, dfQ_individualmodel_concat], ignore_index=True)
    return dfQ_individualmodel_allmodels

# ========================  ADAPTIVE ENSEMBLE TRAJECTORY SELECTION ======================== 

def get_wmape(actual, sim) -> float:
    return np.sum(np.abs(actual - sim)) / np.sum(np.abs(actual))

def computing_wmape_trajs(df, df_surv, list_hor):
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
    for col in df.columns:
        df_col = df[col]
        wmape_result = get_wmape(df_surv[df_surv.horizon.isin(list_hor)]['hospitalizations'].values, df_col.values)
        modelname = col.split('-2024-08-01')[0][:-2]
        dict_wmape_traj[col] = [modelname, wmape_result]
    return dict_wmape_traj

def ranking_trajs(dict_score_traj, toptraj_score, df):
    """
    This function ranks the trajectories based on the loss function scores and selects the top trajectories.
    input:
        dict_wmape_traj = dictionary with WMAPE values for each trajectory
        toptraj_score = percentage of top trajectories to keep
        bigdf = dataframe with all trajectories
    output:
        perc_trajs_scenarios = dictionary with the percentage of trajectories for each scenario
        df_toens = dataframe with the top trajectories
        all_keeptrajs = list of all top trajectories
    """
    scenarios = ['A-2024-08-01', 'B-2024-08-01', 'C-2024-08-01', 'D-2024-08-01', 'E-2024-08-01', 'F-2024-08-01']
    df_score = pd.DataFrame.from_dict(dict_score_traj, orient = 'index', columns=['model_name', 'lossfunction_score'])
    # add column model_name using split 'UT-ImmunoSEIRS" "-D-2024-08-01-17-1'
    #df_score['model_name'] = df_score['ids'].str.split('-2024-08-01').str[0].str[:-2]
    df_topmodels = pd.DataFrame()
    

    # Strategy applied: for each model, take the top x% of trajectories
    for model in list(df_score['model_name'].unique()):
        df_score_model = df_score[df_score.loc[:, 'model_name'] == model].copy()
        df_score_model.sort_values('lossfunction_score', inplace=True)        
        thr = int(toptraj_score * len(df_score_model))
        # Take the best toptraj_score 
        top_trajs = df_score_model.iloc[:thr]
        df_topmodels = pd.concat([df_topmodels, top_trajs], axis = 0)
    df_toens = df[df['ids'].isin(df_topmodels.index)]
    return df_toens



# ========================  SAVE RESULTS  ========================

def concat_with_bootstrapping(meta_dict, label='n_bootstrapping'):
    result_df = pd.DataFrame()
    for n_dwnsample, df in meta_dict.items():
        if isinstance(df, pd.DataFrame):
            df = df.copy()
        elif isinstance(df, dict):
            # we are processing the dictionary of posterior models
            rows = []
            for k_value, dict_model_posterior in df.items():
                for model, posterior_value in dict_model_posterior.items():
                    row = {
                        'k': k_value,
                        'model_name': model,
                        'posterior_value': posterior_value
                    }
                    rows.append(row)
            df = pd.DataFrame(rows)
        #df = df.copy()  # evita SettingWithCopyWarning
        df[label] = n_dwnsample
        result_df = pd.concat([result_df, df], ignore_index=True)
    return result_df

def save_original_bootstrapping(
                dict_original_scenarios_nd,
                dict_original_ens2_nd,
                dict_individual_models_nd,
                state,
                season,
                loss_function,
                path_original,
                path_individualmodels,
                label_scenario):
    # Individual models quantiles
    dfQ_individuals_bootstrapped = concat_with_bootstrapping(dict_individual_models_nd)
    dfQ_individuals_bootstrapped.to_csv(path_individualmodels + f"individual_models_S2_bootstrap_{state}_{loss_function}_{season}_{label_scenario}_k_values.csv", index=False)
    # Original ensemble quantiles for ensemble2
    dfQ_original_ens2_bootstrapped = concat_with_bootstrapping(dict_original_ens2_nd)
    # Original ensemble for single scenarios
    dfQ_original_scenarios_bootstrapped = concat_with_bootstrapping(dict_original_scenarios_nd)
    # concatenate the two dataframes
    dfQ_original_scenarios_bootstrapped = pd.concat([dfQ_original_scenarios_bootstrapped, dfQ_original_ens2_bootstrapped], ignore_index=True)
    dfQ_original_scenarios_bootstrapped.to_csv(path_original + f"original_scenarios_S2_bootstrap_{state}_{loss_function}_{season}_{label_scenario}_k_values.csv", index=False)
    return

def loading_surveillance_adaptive(ref_date, start_date, github_repo, github_directory):
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

    file = f"target-hospital-admissions_{ref_date}.csv"
    file_url = f"https://raw.githubusercontent.com/{github_repo}/main/{github_directory}/{file}"
    df_surv = read_csv_from_github(file_url)
    ref_date = pd.to_datetime(ref_date)
    start_date = pd.to_datetime(start_date)
    df_surv['date'] = pd.to_datetime(df_surv['date'])
    # filter the dataframe to only include rows where the location is 'US' and the date is between start_date and ref_date
    df_surv_date_US = df_surv[(df_surv.location == 'US') & 
                            (df_surv.date <= ref_date) & 
                            (df_surv.date >= start_date)]
    df_surv_date_US = df_surv_date_US.rename(columns={"value": "hospitalizations"})
    df_surv_date_US = df_surv_date_US.sort_values(by='date')
    df_surv_date_US['horizon'] = np.arange(1, len(df_surv_date_US)+1)
    return df_surv_date_US

def main():
    # Define parameters
    github_repo, github_directory, season, loss_function, k_values, num_tot_bootstrap, analysis_type, path_R_script_original, path_R_script, scenarios_ids = define_parameters()
    print("Parameters defined:")
    print(f"Season: {season}, Loss Function: {loss_function}, k_values: {k_values}, Number of Bootstrapping: {num_tot_bootstrap}, Analysis Type: {analysis_type}, Scenarios IDs: {scenarios_ids}")
    
    df_scenarios = load_dataframe_smh(season)
    if len(scenarios_ids) == 1:
        label_scenario = "Sc" + scenarios_ids[0][0]
    else:
        label_scenario = " "
    states_to_process, path_list_trajs, path_adaptive, path_original, path_individualmodels = get_paths_tosave(analysis_type, df_scenarios, season)
    for state in states_to_process:
        print(f'Processing state: {state}')
        df_state_loc = df_scenarios[df_scenarios['location'] == state].copy()
        df_state = df_state_loc[df_state_loc['scenario_id'].isin(scenarios_ids)].copy()
        create_ID_ModelTrajectory(df_state, season)
        df_state, df_surveillance, end_date = load_surveillance_data(season, df_state,  github_repo, github_directory, state)
        # Define all the structures to save the results of boostrapping iterations
        dict_original_scenarios_nd, dict_original_ens2_nd, dict_individual_models_nd = {}, {}, {}
        # Iterate over the number of bootstrapping
        for n_bootstrap in range(num_tot_bootstrap):
            # read json file with trajectories
            with open(f'{path_list_trajs}/list_trajs_bootstrap_wreplacement_{state}_{season}_{n_bootstrap}_{label_scenario}.json', 'r') as f:
                list_trajs = json.load(f)
            
            df_bootstrapped_state = pd.DataFrame(df_state[df_state['ids'].isin(list_trajs)])
            print("Load boostrapped dataframe number: ", n_bootstrap)

            # --------------- Construct ORIGINAL ENSEMBLE with the bootstrapped dataset ---------------
            df_original_scenarios, df_original_ens2 = generate_original_ensembles(df_bootstrapped_state, season, state, path_R_script_original)
            dict_original_scenarios_nd[n_bootstrap] = df_original_scenarios
            dict_original_ens2_nd[n_bootstrap] = df_original_ens2

            # --------- Construct individual models2 quantiles with bootstrapped dataset---------------
            dfQ_individualmodel = individual_model_quantile_computation(df_bootstrapped_state)
            dict_individual_models_nd[n_bootstrap] = dfQ_individualmodel
            # --------- Save all the results of the bootstrapping iterations ---------------
            save_original_bootstrapping(
                dict_original_scenarios_nd,
                dict_original_ens2_nd,
                dict_individual_models_nd,
                state=state,
                season=season,
                loss_function=loss_function,
                path_original=path_original,
                path_individualmodels=path_individualmodels,
                label_scenario=label_scenario)

            #Define period of scenario projections that correspond to the round in the ensemble files
            start_date = datetime(2024, 8, 17)
            end_date = datetime(2025, 5, 31)
            date_list = [(start_date + timedelta(days=x)).strftime("%Y-%m-%d") for x in range(0, (end_date - start_date).days, 7)]
            end_round = len(date_list) #max value of horizon columns for df_scenarios
            round_init = 14 # this is the first round due to surpassing epidemic threshold
            list_hor = list(range(1, round_init + 1)) #to include in the loss function computation also the first data point
            dict_keep_trajs = {}
            for h in df_bootstrapped_state['horizon'].unique().astype(int)[round_init:end_round]:
                list_hor.append(h)
                df_scenario_h = df_bootstrapped_state[df_bootstrapped_state['horizon'].isin(list_hor)]
                ref_date = pd.to_datetime(df_scenario_h.loc[df_scenario_h['horizon'] == h, 'target_end_date'].values[0]) # get reference date for current horizon
                ref_date_surveillance = (ref_date - timedelta(days=7)).date() #survaillance data (7 days before the reference date)
                print(ref_date, ref_date_surveillance)
                if (ref_date_surveillance.strftime('%Y-%m-%d') == '2024-11-23') or (ref_date_surveillance.strftime('%Y-%m-%d') == '2024-12-07') or (ref_date_surveillance.strftime('%Y-%m-%d') == '2025-01-04') or (ref_date_surveillance.strftime('%Y-%m-%d') == '2025-01-18'):
                    ref_date_surveillance = (ref_date).date()
                    df_surv = loading_surveillance_adaptive(ref_date_surveillance, start_date, github_repo, github_directory)
                    df_surv = df_surv.iloc[:-1]
                    print(df_surv)
                else: 
                    df_surv = loading_surveillance_adaptive(ref_date_surveillance, start_date, github_repo, github_directory)
                    print(df_surv)
                new_df = df_scenario_h.pivot(index='horizon', columns='ids', values='value')
                dict_score_traj = computing_wmape_trajs(new_df, df_surv, list_hor)
                for k in k_values:
                    df_toens = ranking_trajs(dict_score_traj, k, df_bootstrapped_state)
                    models_names_k = df_toens.model_name.unique()
                    dfQ_k = quantile_computation(df_toens, models_names_k)
                    dfQ_k_r = pandas_to_r_dataframe(dfQ_k)
                    pandas2ri.activate()
                    r.r.source(path_R_script)
                    dfQ_k_r_r = pandas2ri.py2rpy(dfQ_k_r)
                    day_tosave = ref_date.strftime('%Y-%m-%d')
                    ens_r = r.r['ensemble_lop'](dfQ_k_r_r, h, k, day_tosave, path_adaptive, loss_function, False, "Ens2", n_bootstrap)

                    print(f'Saved results for h: {h}; day_tosave: {day_tosave}')

if __name__ == "__main__":
    main()
