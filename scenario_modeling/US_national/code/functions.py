import pandas as pd
import numpy as np
from functions import *
from rpy2.robjects import pandas2ri
from rpy2 import robjects as r
from datetime import timedelta, datetime
import warnings
import requests
from io import StringIO
from sklearn.metrics import mean_squared_error
# Suppress all warnings temporarily
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

def create_ID_ModelTrajectory(df):
    """
    This function creates a unique identifier for each trajectory in the dataframe.
    The identifier is a combination of the scenario_id, output_type_id and model_name.
    input:
        df = dataframe with all trajectories
    output:
        df = dataframe with a new column that identifies each trajectory
    """
    df['output_type_id'] = df['output_type_id'].astype(int)
    df['output_type_id'] = df['output_type_id'].astype(str)
    df['ids'] = df.apply(lambda x:'%s_%s_%s' % (x['scenario_id'], x['output_type_id'],x['model_name']),axis=1)
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

def loading_surveillance(ref_date, start_date, github_repo, github_directory):
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
    scenarios = ['A-2023-08-14', 'B-2023-08-14', 'C-2023-08-14', 'D-2023-08-14', 'E-2023-08-14', 'F-2023-08-14']
    df_score = pd.DataFrame.from_dict(dict_score_traj, orient='index', columns=['model_name', 'lossfunction_score'])
    df_topmodels = pd.DataFrame()

    # Strategy applied: for each model, take the top x% of trajectories
    for model in list(df_score['model_name'].unique()):
        df_score_model = df_score[df_score.loc[:, 'model_name'] == model]
        df_score_model.sort_values('lossfunction_score', inplace=True)        
        thr = int(toptraj_score * len(df_score_model))
        # Take the best toptraj_score 
        top_trajs = df_score_model.iloc[:thr]
        df_topmodels = pd.concat([df_topmodels, top_trajs], axis = 0)
    df_toens = df[df['ids'].isin(df_topmodels.index)]

    # POSTERIOR DISTRIBUTION ANALYSIS
    perc_trajs_scenarios = {}
    all_keeptrajs = list(df_topmodels.index) # all_keeptrajs is needed for the PERSISTENCE ANALYSIS
    for scenario in scenarios:
        count = sum(traj.startswith(scenario) for traj in all_keeptrajs)
        perc_scenario = count/len(all_keeptrajs) #percentage of trajectories selected from each scenario
        perc_trajs_scenarios[scenario] = perc_scenario
    return perc_trajs_scenarios, df_toens, all_keeptrajs

def computing_rmse(df, df_surv, list_hor):
    """
    This function computes the RMSE for each trajectory in the dataframe.
    input:
        df = dataframe with the trajectories
        df_surv = dataframe with the surveillance data
        list_hor = list of horizons to consider
    output:
        dict_rmse_traj = dictionary with the RMSE values for each trajectory
    """
    dict_rmse_traj = {}
    df = df.iloc[:-1]
    for col in df.columns:
        df_col = df[col]
        rmse = np.sqrt(mean_squared_error(df_surv[df_surv.horizon.isin(list_hor)]['hospitalizations'], df_col.values))
        modelname = col.split('_')[2]
        dict_rmse_traj[col] = [modelname, rmse]
    return dict_rmse_traj

def get_wmape(y_true, y_pred):
    """
    This function computes the Weighted Mean Absolute Percentage Error (WMAPE).
    input:
        y_true = true values
        y_pred = predicted values
    output:
        wmape_result = WMAPE value
    """
    wmape_result = np.abs(y_true - y_pred).sum() / y_true.sum()
    return wmape_result

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
        modelname = col.split('_')[2]
        dict_wmape_traj[col] = [modelname, wmape_result]
    return dict_wmape_traj

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

def pandas_to_r_dataframe(df):
    """
    This function converts a pandas dataframe to an R dataframe.
    input:
        df = dataframe to be converted
    output:
        df = dataframe converted to R dataframe
    """
    return pandas2ri.PandasDataFrame(df)

def jaccard_index(list1, list2):
    """
    This function computes the Jaccard index between two lists.
    input:
        list1 = first list
        list2 = second list
    output:
        jaccard_index = Jaccard index value
    """
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:  # To handle the case where both lists are empty
        return 0.0
    return len(intersection) / len(union)

def compute_jaccard_indices(dict_keep_trajs, mode="to"):
    """
    Compute Jaccard index for trajectory sets.
    
    Parameters:
        dict_keep_trajs (dict): {k_best: list of lists of trajectories}
        mode (str): 'to' to compare to time 0, 'prev' to compare to previous step
    
    Returns:
        dict: {k_best: list of Jaccard index values}
    """
    jaccard_index_dict = {}

    for k_best, traj_list in dict_keep_trajs.items():
        jaccard_list = []

        if mode == "to":
            ref_set = traj_list[0]
            jaccard_list = [jaccard_index(traj, ref_set) for traj in traj_list]
        
        elif mode == "prev":
            jaccard_list = [
                jaccard_index(traj_list[h], traj_list[h - 1])
                for h in range(1, len(traj_list))
            ]
        
        else:
            raise ValueError("Mode must be 'to' or 'prev'.")

        jaccard_index_dict[k_best] = jaccard_list

    return jaccard_index_dict


