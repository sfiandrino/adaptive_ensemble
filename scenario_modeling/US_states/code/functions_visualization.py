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


def get_wis_k_info(df_wis, k):
    df_wis_k = df_wis[df_wis['k_perc'] == k].copy()
    df_wis_k['wis_ratio'] = df_wis_k['wis_adaptive_ensemble2'] / df_wis_k['wis_original_ensemble2']
    ratios_by_state_wis = df_wis_k.groupby('abbreviation')['wis_ratio'].apply(list).to_dict()
    medians_wis = {state: np.median(ratios) for state, ratios in ratios_by_state_wis.items()}
    sorted_states_wis = sorted(medians_wis, key=medians_wis.get)
    ordered_ratios_wis = [ratios_by_state_wis[state] for state in sorted_states_wis]
    return df_wis_k, ratios_by_state_wis, medians_wis, sorted_states_wis, ordered_ratios_wis

def get_colors_wis(medians_wis, sorted_states_wis, cmap):
    norm_wis = plt.Normalize(vmin=np.sqrt(min(medians_wis.values())), vmax=np.sqrt(max(medians_wis.values())))
    colors_wis = [cmap(norm_wis(np.sqrt(medians_wis[state]))) for state in sorted_states_wis]
    return norm_wis, colors_wis

def get_mae_k_info(df_mae, k):
    df_mae_k = df_mae[df_mae['k_perc'] == k].copy()
    df_mae_k['mae_ratio'] = df_mae_k['mae_adaptive_ensemble2'] / df_mae_k['mae_original_ensemble2']
    ratios_by_state_mae = df_mae_k.groupby('abbreviation')['mae_ratio'].apply(list).to_dict()
    medians_mae = {state: np.median(ratios) for state, ratios in ratios_by_state_mae.items()}
    sorted_states_mae = sorted(medians_mae, key=medians_mae.get)
    ordered_ratios_mae = [ratios_by_state_mae[state] for state in sorted_states_mae]
    return df_mae_k, ratios_by_state_mae, medians_mae, sorted_states_mae, ordered_ratios_mae

def get_colors_mae(medians_mae, sorted_states_mae, cmap):
    norm_mae = plt.Normalize(vmin=np.sqrt(min(medians_mae.values())), vmax=np.sqrt(max(medians_mae.values())))
    colors_mae = [cmap(norm_mae(np.sqrt(medians_mae[state]))) for state in sorted_states_mae]
    return norm_mae, colors_mae