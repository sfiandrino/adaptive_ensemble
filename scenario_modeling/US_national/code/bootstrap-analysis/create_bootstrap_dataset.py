import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
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
    scenarios_ids = ['A-2024-08-01', 'B-2024-08-01', 'C-2024-08-01', 'D-2024-08-01', 'E-2024-08-01', 'F-2024-08-01']
    num_tot_bootstrap = 10
    return github_repo, github_directory, season, loss_function, k_values, num_tot_bootstrap, analysis_type, scenarios_ids

def load_dataframe_smh(season):
    df_scenarios = pd.read_parquet("../../../../input_data/SMH_trajectories_FluRound1_2024_2025.parquet")
    df_scenarios.rename(columns={'model_id': 'model_name'}, inplace=True)
    return df_scenarios

def get_paths_tosave(analysis_type, df_scenarios, season):
    if analysis_type == 'US_national':
        states_to_process = ['US']
        path_list_trajs = f"../../output_data/list_trajs_bootstrap/"
    return states_to_process, path_list_trajs

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
    print(df_surv)
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
    print("****")
    print(df_state)
    print("****")
    ref_date = pd.to_datetime(df_state['target_end_date'].max()) # get the last date in the scenarios dataframe
    print(f"Reference date for scenarios: {ref_date}")
    ref_date_surveillance = (ref_date).date()
    print(f"Reference date for surveillance: {ref_date_surveillance}")
    df_surv = loading_surveillance(ref_date_surveillance, start_date, github_repo, github_directory, state)
    return df_state, df_surv, end_date

def len_individual_models(df_state):
    """
    This function computes the length of individual models in the dataframe.
    input:
        df_state = dataframe with all trajectories
    output:
        len_models = list with the length of individual models
    """
    len_models = []
    for model in df_state['model_name'].unique():
        df_model = df_state[df_state['model_name'] == model]
        len_models.append(len(df_model['ids'].unique()))
    min_len = min(len_models)
    return min_len

def construct_bootstrapped_dataset(df_state, min_len, num_scenarios, scenarios_ids):
    df_bootstrapped_state = pd.DataFrame()
    min_len_scenario_id = min_len // num_scenarios

    for model in df_state['model_name'].unique():
        df_model = df_state[df_state['model_name'] == model]
        trajs_selected_scenarios = []

        for scenario_id in scenarios_ids:
            trajs_model_scenario = df_model[df_model['ids'].str.contains(scenario_id)]['ids'].unique().tolist()
            if len(trajs_model_scenario) == 0:
                continue  # Skip if no trajectories for this scenario

            # Bootstrap sampling (with replacement)
            selected_ids = np.random.choice(
                trajs_model_scenario,
                min_len_scenario_id,
                replace=True  # bootstrap mode
            )
            trajs_selected_scenarios.extend(selected_ids)

        df_mod = df_model[df_model['ids'].isin(trajs_selected_scenarios)]
        df_bootstrapped_state = pd.concat([df_bootstrapped_state, df_mod], ignore_index=True)

    return df_bootstrapped_state


def main():
    pandas2ri.activate()
    # Define parameters
    github_repo, github_directory, season, loss_function, k_values, num_tot_bootstrap, analysis_type, scenarios_ids = define_parameters()
    num_scenarios = len(scenarios_ids)
    print("Parameters defined:")
    print(f"Season: {season}, Loss Function: {loss_function}, k_values: {k_values}, Number of Bootstrapping: {num_tot_bootstrap}, Analysis Type: {analysis_type}, Scenarios IDs: {scenarios_ids}")
    df_scenarios_all = load_dataframe_smh(season)
    if len(scenarios_ids) == 1:
        label_scenario = "Sc" + scenarios_ids[0][0]
        print(label_scenario)
    else:
        label_scenario = " "
    df_scenarios = df_scenarios_all[df_scenarios_all['scenario_id'].isin(scenarios_ids)].copy()
    print(df_scenarios.head())
    print(df_scenarios['scenario_id'].unique())
    print(df_scenarios.location.unique())
    states_to_process, path_list_trajs = get_paths_tosave(analysis_type, df_scenarios, season)     
    print(states_to_process)
    for state in states_to_process:
        df_state = df_scenarios[df_scenarios['location'] == state].copy()
        create_ID_ModelTrajectory(df_state, season)
        df_state, df_surv, end_date = load_surveillance_data(season, df_state,  github_repo, github_directory, state)
        # Define len indidivual models to understand the downsampling mode
        min_len = len_individual_models(df_state)
        # Iterate over the number of bootstrapping
        for n_bootstrap in range(num_tot_bootstrap):
            df_downsampled_state = construct_bootstrapped_dataset(df_state, min_len, num_scenarios, scenarios_ids)
            # save in a json file the list of trajectories in downsampled dataset
            list_trajs = df_downsampled_state['ids'].unique().tolist()
            with open(f'../../output_data/review_list_trajs_bootstrap/list_trajs_bootstrap_wreplacement_{state}_{season}_{n_bootstrap}_{label_scenario}.json', 'w') as f:
                json.dump(list_trajs, f)
            print(f"List of trajectories saved for state {state}, bootstrap {n_bootstrap}")
        
if __name__ == "__main__":
    main()