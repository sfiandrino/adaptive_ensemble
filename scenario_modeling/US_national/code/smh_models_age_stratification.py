import pandas as pd
import requests
from io import BytesIO
import numpy as np
import sys
import yaml
import pyarrow.parquet as pq

def loading_trajectories(models_names, submission_date, scenario, target, location, age_group, output_type, is_parquet, github_path, season):
    bigdf = pd.DataFrame()
    for name in models_names:
        folder = name
        parquet_files_ids = list(np.where(np.array(is_parquet) == 1)[0])
        parquet_files = [models_names[i] for i in parquet_files_ids]
        if name in parquet_files:
            github_url = github_path + folder + '/' + submission_date + '-' + folder + '.parquet'
        else:
            github_url = github_path + folder + '/' + submission_date + '-' + folder + '.gz.parquet'
        # scarico il contenuto del file e lo gestisco da parquet a pandas 
        response = requests.get(github_url)
        file_content = BytesIO(response.content)
        parquet_table = pq.read_table(file_content)
        df = parquet_table.to_pandas()
        #Salvo il nome del modello 
        df['model_name'] = [name] * df.shape[0]
        #Seleziono uno scenario
        df = df[df['scenario_id'] == scenario]
        #Seleziono il target
        df = df[df['target'] == target]
        #Seleziono come location US
        df = df[df['location'] == location]
        #Prendo tutti gli age groups
        #Considero solo i sample e non i quantili per poi calcolare la cumulata
        df = df[df['output_type'] == output_type]
        df = df.sort_values(by=['horizon'])
        print('Model: ', name)
        if season == "2024-2025":
            df["stochastic_run"] = df.stochastic_run.astype(int)
            df["traj_id"] = (df["model_name"] + "-" + df["scenario_id"] + "-" + df["stochastic_run"].astype(str)  + "-" +  df["run_grouping"].astype(str)) 
            print("Number of trajectories: ", len(df['traj_id'].unique()))
        else:
            df["output_type_id"] = pd.to_numeric(df["output_type_id"])
            print("Number of trajectories: ", len(df['output_type_id'].unique()))
        print("Number of age groups:", df['age_group'].unique())
    return bigdf

if __name__ == "__main__":
    season = '2024-2025'
    if len(sys.argv) != 0:
        with open("../../../input_data/input_parameters.yml", "r") as f:
            input_params = list(yaml.safe_load_all(f))
        models_names = input_params[0]['models_names']
        submission_date = input_params[0]['submission_date']
        scenarios = input_params[0]['scenarios']
        target = input_params[0]['target']
        location = input_params[0]['location']
        age_group = input_params[0]['age_group']
        output_type = input_params[0]['output_type']
        is_parquet = input_params[0]['is_parquet']
        github_path = input_params[0]['github_path']
        print(output_type)
        final_df = pd.DataFrame()
        for scenario in scenarios:
            print('Scenario: ', scenario)
            loading_trajectories(models_names, submission_date, scenario, target, location, age_group, output_type, is_parquet, github_path, season)
        




