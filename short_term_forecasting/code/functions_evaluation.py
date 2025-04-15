import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import requests
from io import StringIO
from datetime import timedelta, datetime

def get_perc_error(actual, sim, last=4): 
    return 100 * np.mean(np.abs(actual[-last:] - sim[-last:]) / actual[-last:])

def get_wmape(actual, sim) -> float:
    return np.sum(np.abs(actual - sim)) / np.sum(np.abs(actual))

def diff(a, b, norm=False): 
    if norm:
        if a != 0:
            return (a - b) / a
        else: 
            return 0
    else:
        return (a - b)

def interval_score(y, u, l, alpha, norm=False): 
    return -diff(l, u, norm=norm) + 2 / alpha * -diff(y, l, norm=norm) * (y < l) + 2 / alpha * diff(y, u, norm=norm) * (y > u)
    

def weighted_interval_score(y, u_s, l_s, m, alpha_ks, w0=1./2., norm=False):
    K = len(alpha_ks)
    wks = np.array(alpha_ks) / 2.
    return 1. / (K + 1./2.) * (w0 * np.abs(diff(y, m, norm=norm)) + np.dot(wks, [interval_score(y, u, l, a_k, norm=norm) for u, l, a_k in zip(u_s, l_s, alpha_ks)]))

def get_upper_bound(sim_stats, alpha, idx): 
    q2 = 1.0 - alpha / 2
    return sim_stats.loc[np.isclose(sim_stats["quantile"], q2), "value"].values[idx]

def get_lower_bound(sim_stats, alpha, idx): 
    q1 = alpha / 2.
    return sim_stats.loc[np.isclose(sim_stats["quantile"], q1), "value"].values[idx]

def loading_surveillance_eval(start_date, github_repo, github_directory, surveillance_file, state, horizon_to_start):
    """
    This function loads the surveillance data from a GitHub repository.
    input:
        start_date = start date for the data
        github_repo = GitHub repository name
        github_directory = directory in the GitHub repository
        surveillance_file = name of the surveillance file
        state = state to filter the data
        horizon_to_start = horizon to start the data
    output:
        df_surv_date_US = dataframe with the filtered surveillance data
    """
    file_url = f"https://raw.githubusercontent.com/{github_repo}/main/{github_directory}/{surveillance_file}"
    df_surv = read_csv_from_github(file_url)
    start_date = pd.to_datetime(start_date)
    df_surv['date'] = pd.to_datetime(df_surv['date'])
    df_surv_date_US = df_surv[(df_surv.location == state) & 
                            (df_surv.date >= start_date)]
    df_surv_date_US = df_surv_date_US.rename(columns={"value": "hospitalizations"})
    df_surv_date_US = df_surv_date_US.sort_values(by='date')
    df_surv_date_US['horizon'] = np.arange(1, len(df_surv_date_US)+1)
    df_surv_date_US.drop(columns=['X'], inplace=True)
    df_surv_date_US = df_surv_date_US[df_surv_date_US['horizon'] >= horizon_to_start]
    df_surv_date_US['date'] = pd.to_datetime(df_surv_date_US['date'])
    return df_surv_date_US

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

def import_ensemble2_forecast(file_path, surv_lookup, horizon_to_date):
    df = pd.read_csv(file_path, index_col=0)
    df['date'] = df['horizon'].map(horizon_to_date)
    df['date'] = pd.to_datetime(df['date'])
    df['horizon'] = df['horizon'].replace(
            {h: i for i, h in enumerate(df.horizon.unique())}
        )
    df.rename(columns={"quantiles": "output_type_id"}, inplace=True)
    return df.merge(surv_lookup, on='date', how='left')   

def import_baseline(date):
    filename = f"{date}-FluSight-baseline.csv"
    # Correct URL to fetch the raw content from GitHub
    url = f"https://raw.githubusercontent.com/cdcepi/Flusight-baseline/main/Archive_2324/FluSight Baseline Forecasts/{filename}"
    df = read_csv_from_github(url)
    return df

def import_ens_flusight(date):
    filename = f"{date}-FluSight-ensemble.csv"
    # Correct URL to fetch the raw content from GitHub
    url = f"https://raw.githubusercontent.com/cdcepi/Flusight-ensemble/main/Archive_2324/FluSight-ensemble/{filename}"
    df = read_csv_from_github(url)
    return df

def compute_dict_WIS_AE_forecast(df):
    dict_wis_horizon = {}
    dict_ae_horizon = {}
    for horizon in df['horizon'].unique():
        df_horizon = df[df['horizon'] == horizon]
        df_horizon['output_type_id'] = pd.to_numeric(df_horizon['output_type_id'], errors='coerce')
        quantiles = df_horizon['output_type_id'].unique()
        quantile_values = df_horizon.pivot(index='date', columns='output_type_id', values='value')
        u_s = quantile_values.loc[:, quantiles[quantiles > 0.5]].values
        l_s = quantile_values.loc[:, quantiles[quantiles < 0.5]].values[:, ::-1]
        m = quantile_values[0.5].values
        alpha_ks = 1 - 2 * np.abs(quantiles[quantiles < 0.5])
        wis_values = []
        for i in range(len(m)):
            y = df_horizon.hospitalizations.values[0]
            wis = weighted_interval_score(y, u_s[i], l_s[i], m[i], alpha_ks)
            wis_values.append(wis)
        wmape_median = get_wmape(df_horizon.hospitalizations.values[0], m[0])
        mean_wis = np.mean(wis_values)
        dict_wis_horizon[horizon] = mean_wis
        dict_ae_horizon[horizon] = wmape_median
    return dict_wis_horizon, dict_ae_horizon

def coverage(grp, q1, q2):
    """
    Computes the coverage of the forecast.
    input:
        grp = group of data
        q1 = lower quantile
        q2 = upper quantile
    output:
        1 if the forecast is covered, 0 otherwise
    """
    lower_val = grp[grp.output_type_id_str == q1].value.values[0]
    upper_val = grp[grp.output_type_id_str == q2].value.values[0]
    if grp.hospitalizations.values[0] >= lower_val and grp.hospitalizations.values[0] <= upper_val:
        return 1 
    else:
        return 0
