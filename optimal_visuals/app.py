import faicons as fa

# Load data and compute static values
from shiny import reactive, render
from shiny.express import input, ui

from plotly.subplots import make_subplots
import plotly.graph_objects as go
# import plotly.express as px
from shinywidgets import render_plotly

# import numpy as np
# from scipy.spatial import Voronoi, voronoi_plot_2d
# import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def remove_base_case_and_non_diffs(df, remove_base = True, remove_nondiffs = True):
    '''
    Removes the base case and original output columns from df for the purposes of heatmap visualizations and interval calculations
    '''
    if remove_nondiffs:
        diff_columns = df.columns.map(lambda x: ("diff" in x) or x=="sensitivity" or x=="threshold")
        df.loc[:, diff_columns] = df
      
    if remove_base:
        df = df.loc[(df['threshold'] > 0), :]
        df['sensitivity'] = df['sensitivity'].astype(pd.api.types.CategoricalDtype(categories = ['high', 'mid', 'low'], ordered = True))

        # Remove the 'none' group from the ordered categorical variable so that it doesn't appear in any df.groupby() results or in heatmaps
        # df['sensitivity'] = df['sensitivity'].astype(pd.api.types.CategoricalDtype(categories = ['high', 'mid', 'low'], ordered = True))
    return df

app_dir = Path(__file__).parent
all_data = pd.read_parquet(app_dir / "all_data.parquet")
psc_data = pd.read_parquet(app_dir / "psc_data.parquet")
map_data = pd.read_parquet(app_dir / "maps.parquet")

# Add page title and sidebar
ui.page_opts(title="Stroke Simulations", fillable=True)

with ui.sidebar(open="desktop"):
    ui.input_slider(
        id = 'equipoise_range',
        label = 'Equipoise range',
        min = 0,
        max = 1,
        value = [0, 1]
    )

    ui.input_slider(
        id = 'geoscale_range',
        label = 'Geoscale range',
        min = 30,
        max = 100,
        value = [30, 100]
    )

    ui.input_checkbox_group(
        "sensitivity",
        "LVO diagnosis sensitivity and specificity",
        ["0.9, 0.6", "0.75, 0.75", "0.6, 0.9"],
        selected = ["0.9, 0.6", "0.75, 0.75", "0.6, 0.9"]
    )
    ui.input_switch(
        "psc_only",
        "Only patients closest to PSC",
        )
    
    ui.input_action_button("reset", "Reset filter")
    ui.input_dark_mode(
        id = 'dark_mode',
        mode = 'light'
    )

# Add main content
ICONS = {
    "square": fa.icon_svg("square", "regular"),
    "ruler": fa.icon_svg("ruler"),
    "map": fa.icon_svg("map", "regular")
}

with ui.layout_columns(fill=False):
    with ui.value_box(showcase=ICONS["map"]):
        "Number of maps"

        @render.express
        def get_num_maps():
            get_data()['map'].unique().shape[0]

    with ui.value_box(showcase=ICONS["square"]):
        "Average equipoise"

        @render.express
        def get_avg_equipoise():
            f'{map_data.loc[map_data['map'].isin(get_data()['map'].unique()), 'equipoise'].mean():.3f}'

    with ui.value_box(showcase=ICONS["ruler"]):
        "Average length"

        @render.express
        def get_avg_geoscale():
            f'{map_data.loc[map_data['map'].isin(get_data()['map'].unique()), 'geoscale'].mean():.3f}'

#     with ui.value_box(showcase=ICONS["ruler"]):
#         "Map side length"

#         @render.express
#         def get_geoscale():
#             if input.map_number() is None or input.map_number() < 0:
#                 '--'
#             else:
#                 f'{map_data.loc[map_data['map'] == input.map_number(), 'geoscale'].values.item(0):.3f}'

with ui.layout_columns(col_widths=[4,4,4]):

    with ui.card(full_screen=True):
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Change in mRS for ischemic patients"

        #@render.plot
        @render_plotly
        def mRS_ischemic_plots():
            var_name = 'ischemic_patients_diff'
            scenarios = get_optimal_scenarios(var_name, 'max')
            
            sensitivity_counts = scenarios['sensitivity'].value_counts().fillna(0).sort_index()
            threshold_counts = scenarios['threshold'].value_counts().sort_index()
            
            fig = make_subplots(rows = 2, cols = 1)

            fig.add_trace(go.Bar(x = sensitivity_counts.index.tolist(), y = sensitivity_counts.values),
                          row = 1, col = 1)

            fig.add_trace(go.Bar(x = threshold_counts.index.tolist(), y = threshold_counts.values),
                          row = 2, col = 1)
            # ax = sns.lineplot(data, x = 'threshold', y = 'lvo_patients_diff', hue = 'sensitivity', marker = 'o', errorbar = None)
            # return ax
            # return px.line(data, x = 'threshold', y = 'lvo_patients_diff', color = 'sensitivity', markers = True)
            fig.update_layout(showlegend = False)
            return fig

        
    with ui.card(full_screen=True):
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Change in mRS for LVO patients"

        #@render.plot
        @render_plotly
        def mRS_lvo_plot():
            var_name = 'lvo_patients_diff'
            scenarios = get_optimal_scenarios(var_name, 'max')
            
            sensitivity_counts = scenarios['sensitivity'].value_counts().fillna(0).sort_index()
            threshold_counts = scenarios['threshold'].value_counts().sort_index()
            
            fig = make_subplots(rows = 2, cols = 1)

            fig.add_trace(go.Bar(x = sensitivity_counts.index.tolist(), y = sensitivity_counts.values),
                          row = 1, col = 1)

            fig.add_trace(go.Bar(x = threshold_counts.index.tolist(), y = threshold_counts.values),
                          row = 2, col = 1)
            
            fig.update_layout(showlegend = False)
            # ax = sns.lineplot(data, x = 'threshold', y = 'lvo_patients_diff', hue = 'sensitivity', marker = 'o', errorbar = None)
            # return ax
            # return px.line(data, x = 'threshold', y = 'lvo_patients_diff', color = 'sensitivity', markers = True)
            return fig
        
    with ui.card(full_screen=True):
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Change in Time to EVT for LVO patients"

            #@render.plot
        @render_plotly
        def time_evt_lvo_plot():
            var_name = 'evt_lvo_mean_diff'
            scenarios = get_optimal_scenarios(var_name, 'min')
            
            sensitivity_counts = scenarios['sensitivity'].value_counts().fillna(0).sort_index()
            threshold_counts = scenarios['threshold'].value_counts().sort_index()
            
            fig = make_subplots(rows = 2, cols = 1)

            fig.add_trace(go.Bar(x = sensitivity_counts.index.tolist(), y = sensitivity_counts.values),
                          row = 1, col = 1)

            fig.add_trace(go.Bar(x = threshold_counts.index.tolist(), y = threshold_counts.values),
                          row = 2, col = 1)
            # ax = sns.lineplot(data, x = 'threshold', y = 'lvo_patients_diff', hue = 'sensitivity', marker = 'o', errorbar = None)
            # return ax
            # return px.line(data, x = 'threshold', y = 'lvo_patients_diff', color = 'sensitivity', markers = True)
            fig.update_layout(showlegend = False)
            return fig



ui.include_css(app_dir / "styles.css")

# --------------------------------------------------------
# Reactive calculations and effects
# --------------------------------------------------------

@reactive.calc
def get_data():
    if input.psc_only():
        data = psc_data
    else:
        data = all_data
    data = remove_base_case_and_non_diffs(data, remove_nondiffs = True)

    sensitivity_dict = {
        '0.9, 0.6': 'high',
        '0.75, 0.75': 'mid',
        '0.6, 0.9': 'low'
    }
    test_sensitivities = [sensitivity_dict[i] for i in input.sensitivity()]
    data = data.loc[data['sensitivity'].isin(test_sensitivities), :]
    map_nums = map_data.loc[(map_data['equipoise'].between(input.equipoise_range()[0], input.equipoise_range()[1])) & (map_data['geoscale'].between(input.geoscale_range()[0], input.geoscale_range()[1])), 'map'].unique()
    data = data.loc[data['map'].isin(map_nums),:].copy()
    return data
    # lvo_sensivity 


def get_optimal_scenarios(var_str, max_or_min):
    data = get_data().loc[:, ['map', 'sensitivity', 'threshold', var_str]]
    data = data.groupby(['map','sensitivity','threshold']).mean().reset_index('map')
    match max_or_min:
        case 'max':
            scenarios = data.groupby('map').idxmax()
        case 'min':
            scenarios = data.groupby('map').idxmin()
    scenarios[['sensitivity', 'threshold']] = pd.DataFrame(scenarios[var_str].tolist(), index = scenarios.index)
    scenarios.drop(var_str, axis = 1, inplace = True)
    scenarios['sensitivity'] = scenarios['sensitivity'].astype(pd.api.types.CategoricalDtype(categories = ['low', 'mid', 'high'], ordered = True))
    return scenarios

@reactive.effect
@reactive.event(input.reset)
def _():
    # ui.update_slider("total_bill", value=bill_rng)
    # ui.update_checkbox_group("time", selected=["Lunch", "Dinner"])
    ui.update_checkbox_group(
        id = "sensitivity",
        selected = ["0.9, 0.6", "0.75, 0.75", "0.6, 0.9"]
    )
    ui.update_slider(
        'equipoise_range',
        value = [0, 1]
    )
    ui.update_slider(
        'geoscale_range',
        value = [30, 100]
    )

    ui.update_switch(id = "psc_only", value = False)