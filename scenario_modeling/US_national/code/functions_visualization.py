import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import requests
from io import StringIO
from datetime import timedelta, datetime
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
from functions_visualization import *
font_path = "font-folder/Roboto-Regular.ttf"
font_path_bold = "font-folder/Roboto-Bold.ttf"
font_prop = fm.FontProperties(fname=font_path, size=18)
font_prop_legend = fm.FontProperties(fname=font_path, size=16)
font_prop_title = fm.FontProperties(fname=font_path_bold, size=22)
import matplotlib as mpl
mpl.rcParams['font.family'] = font_prop.get_name()
import warnings
warnings.filterwarnings('ignore')


def get_wis_dataframes(df_wis, df_surv):
    """
    Function to get the WIS dataframes for different k_percentiles and merge with hospitalizations data.
    Args:
        df_wis (pd.DataFrame): Dataframe containing WIS data.
        df_surv (pd.DataFrame): Dataframe containing hospitalizations data.
    Returns:
        tuple: Dataframes for k_percentiles 5, 15, 25, 50, and 75.
    """
    df_wis_k_5 = df_wis[df_wis['k_perc'] == 5]
    df_wis_k_15 = df_wis[df_wis['k_perc'] == 15]
    df_wis_k_25 = df_wis[df_wis['k_perc'] == 25]
    df_wis_k_50 = df_wis[df_wis['k_perc'] == 50]
    df_wis_k_75 = df_wis[df_wis['k_perc'] == 75]
    df_wis_k_5['week'] = pd.to_datetime(df_wis_k_5['week'])
    df_wis_k_15['week'] = pd.to_datetime(df_wis_k_15['week'])
    df_wis_k_25['week'] = pd.to_datetime(df_wis_k_25['week'])
    df_wis_k_50['week'] = pd.to_datetime(df_wis_k_50['week'])
    df_wis_k_75['week'] = pd.to_datetime(df_wis_k_75['week'])
    df_surv['week'] = pd.to_datetime(df_surv['week'])
    df_wis_k_25 = pd.merge(left=df_wis_k_25, right=df_surv[['week', 'hospitalizations']], how='left', on="week")
    df_wis_k_25 = df_wis_k_25[['week', 'wis_adaptive_ensemble2', 'wis_original_ensemble2', 'wis_rel_original2', 'hospitalizations']]
    return df_wis_k_5, df_wis_k_15, df_wis_k_25, df_wis_k_50, df_wis_k_75

def get_mae_dataframes(df_mae, df_surv):
    """
    Function to get the MAE dataframes for different k_percentiles and merge with hospitalizations data.
    Args:
        df_mae (pd.DataFrame): Dataframe containing MAE data.
        df_surv (pd.DataFrame): Dataframe containing hospitalizations data.
    Returns:
        tuple: Dataframes for k_percentiles 5, 15, 25, 50, and 75.
    """
    df_mae_k_5 = df_mae[df_mae['k_perc'] == 5]
    df_mae_k_15 = df_mae[df_mae['k_perc'] == 15]
    df_mae_k_25 = df_mae[df_mae['k_perc'] == 25]
    df_mae_k_50 = df_mae[df_mae['k_perc'] == 50]
    df_mae_k_75 = df_mae[df_mae['k_perc'] == 75]
    df_mae_k_5['week'] = pd.to_datetime(df_mae_k_5['week'])
    df_mae_k_15['week'] = pd.to_datetime(df_mae_k_15['week'])
    df_mae_k_25['week'] = pd.to_datetime(df_mae_k_25['week'])
    df_mae_k_50['week'] = pd.to_datetime(df_mae_k_50['week'])
    df_mae_k_75['week'] = pd.to_datetime(df_mae_k_75['week'])
    df_surv['week'] = pd.to_datetime(df_surv['week'])
    df_mae_k_25 = pd.merge(left=df_mae_k_25, right=df_surv[['week', 'hospitalizations']], how='left', on="week")
    df_mae_k_25 = df_mae_k_25[['week', 'mae_adaptive_ensemble2', 'mae_original_ensemble2', 'mae_rel_original2', 'hospitalizations']]
    return df_mae_k_5, df_mae_k_15, df_mae_k_25, df_mae_k_50, df_mae_k_75

def plot_scenario_polar(df, k_value, output_path, font_prop=None, font_prop_title=None):
    df_k = df[df['k'] == k_value]
    df_k_pivot = df_k.pivot(index='week', columns='scenario', values='posterior_value').reset_index()
    scenarios = df_k_pivot.columns[1:].tolist()

    vmin = df_k_pivot[[c for c in df_k_pivot.columns if c != "week"]].min().min()
    vmax = df_k_pivot[[c for c in df_k_pivot.columns if c != "week"]].max().max()

    df_k_pivot['date_month'] = pd.to_datetime(df_k_pivot['week']).dt.strftime('%Y-%m')
    df_k_pivot['date_week'] = pd.to_datetime(df_k_pivot['week']).dt.strftime('%y-%W')

    df_year_month = df_k_pivot.groupby(['date_month']).mean(numeric_only=True).reset_index()

    vmin = df_year_month[[c for c in df_year_month.columns if c != "date_month"]].min().min()
    vmax = df_year_month[[c for c in df_year_month.columns if c != "date_month"]].max().max()

    max_per_week = df_k_pivot[scenarios].max(axis=1)
    colors = ['#c4e6c3', '#80c799', '#4da284', '#1d4f60']
    cmap = mpl.colors.LinearSegmentedColormap.from_list("my color", colors, N=256)
    norm = mpl.colors.Normalize(vmin=0, vmax=df_k_pivot[scenarios].max().max())

    fig, axes = plt.subplots(figsize=(9, 12.6), subplot_kw={"projection": "polar"}, 
                             ncols=2, nrows=3, dpi=300)
    angles = np.linspace(0.05, 2 * np.pi - 0.05, len(df_k_pivot), endpoint=False)[::-1]
    for scenario, ax in zip(scenarios, axes.ravel()):
        lengths = df_k_pivot[scenario].values
        ax.set_theta_offset(np.pi / 2 * 1.18)
        ax.set_ylim(0, max(lengths) + 0.06)
        for angle, length, max_value in zip(angles, lengths, max_per_week):
            edge_color = "#103a43" if length == max_value else "k"
            linewidth = 1.5 if length == max_value else 0.3
            ax.bar(angle, length, color=cmap(norm(length)), alpha=1., width=0.25,
                   linewidth=linewidth, edgecolor=edge_color)
        ax.set_xticks(angles)
        ax.set_xticklabels(df_k_pivot["date_week"], size=8, color="k")
        ax.xaxis.set_tick_params(pad=8)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4], size=8, color="dimgray",
                           fontproperties=font_prop_title)
        ax.grid(linewidth=0.1, color="grey")

        for label in ax.get_yticklabels() + ax.get_xticklabels():
            label.set_fontproperties(font_prop)

        ax.set_title(f"Scenario {scenario.split('-')[0]}", fontproperties=font_prop_title)

    plt.tight_layout()
    filename = f"{output_path}/scenario_posterior_k{k_value}_main.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    print(f"Plot for k={k_value} saved in in {filename}")
