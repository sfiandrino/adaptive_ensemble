import pandas as pd
import requests
from io import BytesIO
import numpy as np
import sys
import yaml
import pyarrow.parquet as pq
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
from rpy2 import robjects as r

def create_ID_ModelTrajectory(df):
    """
    This function creates a unique identifier for each trajectory in the dataframe.
    The identifier is a combination of the scenario_id, output_type_id and model_name.
    input:
        df = dataframe with all trajectories
    output:
        df = dataframe with a new column that identifies each trajectory
    """
    df['output_type_id'] = df['output_type_id'].astype(str)
    df['ids'] = df.apply(lambda x:'%s_%s_%s' % (x['scenario_id'], x['output_type_id'],x['model_name']),axis=1)
    return df

def quantile_computation(df):
    """
    This function computes quantiles for each model and horizon in the dataframe.
    input:
        df = dataframe with all trajectories
    output:
        dfQ = dataframe with quantiles for each model and horizon
    """
    models_names = df['model_name'].unique()
    quantiles = np.concatenate(([0.01, 0.025], np.arange(0.05, 1, 0.05), [0.975, 0.99]))
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

def computing_ensemble(dfQ_r, scenario, path_R_script, day_to_save, path_to_save, loss_function, is_original, hor, top_traj, state):
    """
    This function computes the ensemble using the R script.
    input:
        dfQ_r = dataframe with quantiles for each model and horizon
        state = state for which the ensemble is computed
        scenario = scenario for which the ensemble is computed
    output:
        ens_r = ensemble computed by the R script
    """
    pandas2ri.activate()
    # load the R script
    r.r.source(path_R_script)
    dfQ_r_r = pandas2ri.py2rpy(dfQ_r)
    if scenario is not "Ens2":
        scenario = scenario[0]
    ens_r = r.r['ensemble_lop'](dfQ_r_r, hor, top_traj, day_to_save, path_to_save, loss_function, is_original, scenario, state)
    return ens_r

if __name__ == "__main__":
    path_R_script = "/home/sfiandrino/PhD_Project/adaptive_ensemble_methodological/scenario_modeling/US_states/code/ensemble_lop.r"
    path_to_save = "../output_data/original_ensembles/"
    # for the original ensemble the following data entries for the ensemble lop function are not used
    day_to_save = " "
    loss_function = " "
    hor = ""
    top_traj = " "
    # flag to indicate we are generating the original ensemble
    is_original = True
    # Load the data
    df_scenarios = pd.read_csv("../../../input_data/SMH_trajectories_FluRound1_2023_2024_states.csv", index_col = 0)
    # remove state = 'US' - national analysis has already been done
    df_scenarios = df_scenarios[df_scenarios['location'] != 'US']
    # Generate original ensemble for single scenarios
    for state in df_scenarios.location.unique():
        df_scenarios_state = df_scenarios[df_scenarios.loc[:, 'location'] == state]
        for scenario in df_scenarios_state.scenario_id.unique():
            df_scenario = df_scenarios_state[df_scenarios_state.loc[:, 'scenario_id'] == scenario]
            create_ID_ModelTrajectory(df_scenario)
            dfQ = quantile_computation(df_scenario)
            dfQ_r = pandas_to_r_dataframe(dfQ)
            ens_r = computing_ensemble(dfQ_r, scenario, path_R_script, day_to_save, path_to_save, loss_function, is_original, hor, top_traj, state)
        # Generate original ensemble for the Ensemble2
        create_ID_ModelTrajectory(df_scenarios_state)
        dfQ = quantile_computation(df_scenarios_state)
        dfQ_r = pandas_to_r_dataframe(dfQ)
        ens_r = computing_ensemble(dfQ_r, "Ens2", path_R_script, day_to_save, path_to_save, loss_function, is_original, hor, top_traj, state)





