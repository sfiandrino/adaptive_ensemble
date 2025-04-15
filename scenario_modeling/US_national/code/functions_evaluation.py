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
    # get upper bound levels
    q2 = 1.0 - alpha / 2
    return sim_stats.loc[sim_stats["quantile"] == q2]["value"].values[idx]

def get_lower_bound(sim_stats, alpha, idx): 
    # get lower bound levels
    q1 = alpha / 2.
    return sim_stats.loc[sim_stats["quantile"] == q1]["value"].values[idx]


def get_aggregate_wis(realdata, 
                      sim_stats, 
                      alphas=[0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90], 
                      aggr_fun=None, 
                      norm=False):
    wis = []
    for i in range(len(realdata)):
        wis.append(weighted_interval_score(
                            y=realdata[i], 
                            u_s=np.array([get_upper_bound(sim_stats, alpha=alpha, idx=i) for alpha in alphas]), 
                            l_s=np.array([get_lower_bound(sim_stats, alpha=alpha, idx=i) for alpha in alphas]), 
                            m=sim_stats.loc[sim_stats["quantile"] == 0.5]["value"].values[i],
                            alpha_ks=alphas, 
                            norm=norm))
    if aggr_fun != None:
        return aggr_fun(wis)
    else:
        return wis

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

def compute_dict_WIS_AE(df, alphas):
    if 'quantiles' in df.columns:
        df = df.rename(columns={"quantiles": "quantile"})
    realdata = df[df['quantile'] == 0.5]['hospitalizations'].values
    sim_stats = df.drop(columns=['date', 'hospitalizations', 'horizon'])
    wmape = get_wmape(realdata, sim_stats.loc[sim_stats["quantile"] == 0.5]["value"].values)
    wis_list = get_aggregate_wis(realdata, 
                      sim_stats, 
                      alphas=alphas, 
                      aggr_fun=None, 
                      norm=False)
    wis_mean = np.mean(wis_list)
    return wis_list, wis_mean, wmape

def import_ensemble2(file_path, surv_lookup, horizon_to_date):
    df = pd.read_csv(file_path, index_col=0)
    df['date'] = df['horizon'].map(horizon_to_date)
    df['date'] = pd.to_datetime(df['date'])
    return df.merge(surv_lookup, on='date', how='left')   

def import_ensemble_original(file_path, df_surv, date_ref):
    df_surv['date'] = pd.to_datetime(df_surv['date'])
    df = pd.read_csv(file_path, index_col=0)
    valid_horizons = df_surv['horizon'].unique()
    df = df[df['horizon'].isin(valid_horizons)]
    # Map each horizon to its corresponding date
    horizon_to_date = df_surv.set_index('horizon')['date'].to_dict()
    df['date'] = df['horizon'].map(horizon_to_date)
    # Filter to only include dates after or equal to date_only
    date_ref = pd.to_datetime(date_ref)
    df = df[df['date'] >= date_ref]
    # Merge in hospitalization targets
    df = df.merge(df_surv[['date', 'hospitalizations']].drop_duplicates(), on='date', how='left')
    return df