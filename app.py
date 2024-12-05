import faicons as fa

# Load data and compute static values
from shiny import reactive, render
from shiny.express import input, ui

import plotly.express as px
from shinywidgets import render_plotly

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import pandas as pd

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

all_data = pd.read_parquet("../all_data.parquet")
psc_data = pd.read_parquet("../psc_data.parquet")
map_data = pd.read_parquet("../maps.parquet")

# Add page title and sidebar
ui.page_opts(title="Stroke Simulations", fillable=True)

with ui.sidebar(open="desktop"):
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
    ui.input_numeric(
        'map_number',
        label = 'Map number',
        value = None
    )
    ui.input_action_button("reset", "Reset filter")
    ui.input_dark_mode(
        id = 'dark_mode',
        mode = 'light'
    )

# Add main content
ICONS = {
    "square": fa.icon_svg("square", "regular"),
    "ruler": fa.icon_svg("ruler")
}

with ui.layout_columns(fill=False):
    with ui.value_box(showcase=ICONS["square"]):
        "Equipoise"

        @render.express
        def get_equipoise():
            if input.map_number() is None or input.map_number() < 0:
                '--'
            else:
                f'{map_data.loc[map_data['map'] == input.map_number(), 'equipoise'].values.item(0):.3f}'

    with ui.value_box(showcase=ICONS["ruler"]):
        "Map side length"

        @render.express
        def get_geoscale():
            if input.map_number() is None or input.map_number() < 0:
                '--'
            else:
                f'{map_data.loc[map_data['map'] == input.map_number(), 'geoscale'].values.item(0):.3f}'

with ui.layout_columns(col_widths=[6, 6, 6, 6]):
    with ui.card(full_screen=True):
        ui.card_header("Map visualization")

        @render.plot
        def show_map():
            # return render.DataGrid(get_data())
            if input.map_number() is None or input.map_number() < 0:
                return None
            geoscale, xPSC, yPSC, xPSC2, yPSC2 = map_data.loc[map_data['map'] == input.map_number(), ['geoscale', 'xPSC', 'yPSC', 'xPSC2', 'yPSC2']].values.flatten()

            med_coords = np.array([[0.5 * geoscale, 0.5 * geoscale],
                        [xPSC, yPSC],
                        [xPSC2, yPSC2]])
            
            coord_labels = ['CSC', 'PSC', 'PSC2']
            voronoi_colors = ['blue', 'green', 'red']
            voronoi_markers = ['^','o','o']

            distant_coords = np.array([[-8 * geoscale, -8 * geoscale],
                           [8 * geoscale, 8 * geoscale],
                           [-8 * geoscale, 9 * geoscale],
                           [8 * geoscale, -7 * geoscale]])

            full_coords = np.vstack((med_coords, distant_coords))
            vor = Voronoi(full_coords)
            voronoi_plot_2d(vor, show_vertices = False, show_points = False)
            for i, hosp in enumerate(coord_labels):
                poly = [vor.vertices[j] for j in vor.regions[vor.point_region[i]]]
                plt.fill(*zip(*poly), color = voronoi_colors[i], alpha = 0.25)
                plt.scatter(med_coords[i,0], med_coords[i,1], c = voronoi_colors[i], label = hosp, marker = voronoi_markers[i])
            plt.xlim(0, geoscale)
            plt.ylim(0, geoscale)
            plt.legend()
            plt.gca().set_aspect('equal')
            return plt.gca()
            

    with ui.card(full_screen=True):
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Change in mRS for ischemic patients"

        #@render.plot
        @render_plotly
        def mRS_ischemic_plot():
            data = get_data().groupby(['sensitivity','threshold']).mean().reset_index()
            # ax = sns.lineplot(data, x = 'threshold', y = 'ischemic_patients_diff', hue = 'sensitivity', marker = 'o', errorbar = None)
            #return ax
            return px.line(data, x = 'threshold', y = 'ischemic_patients_diff', color = 'sensitivity', markers = True)


    with ui.card(full_screen=True):
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Change in Time to EVT for LVO patients"

            #@render.plot
        @render_plotly
        def time_evt_lvo_plot():
            data = get_data().groupby(['sensitivity','threshold']).mean().reset_index()
            # ax = sns.lineplot(data, x = 'threshold', y = 'evt_lvo_mean_diff', hue = 'sensitivity', marker = 'o', errorbar = None)
            # return ax
            return px.line(data, x = 'threshold', y = 'evt_lvo_mean_diff', color = 'sensitivity', markers= True)

        
    with ui.card(full_screen=True):
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Change in mRS for LVO patients"

        #@render.plot
        @render_plotly
        def mRS_lvo_plot():
            data = get_data().groupby(['sensitivity','threshold']).mean().reset_index()
            # ax = sns.lineplot(data, x = 'threshold', y = 'lvo_patients_diff', hue = 'sensitivity', marker = 'o', errorbar = None)
            # return ax
            return px.line(data, x = 'threshold', y = 'lvo_patients_diff', color = 'sensitivity', markers = True)



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
    data = data.loc[data['sensitivity'].isin(test_sensitivities), :].copy()
    if input.map_number() is not None and input.map_number() > 0:
        data = data.loc[data['map'] == input.map_number(), :].copy()
    return data
    # lvo_sensivity 

@reactive.effect
@reactive.event(input.reset)
def _():
    # ui.update_slider("total_bill", value=bill_rng)
    # ui.update_checkbox_group("time", selected=["Lunch", "Dinner"])
    ui.update_checkbox_group(
        id = "sensitivity",
        selected = ["0.9, 0.6", "0.75, 0.75", "0.6, 0.9"]
    )
    ui.update_switch(id = "psc_only", value = False)
    ui.update_numeric(
        id = 'map_number',
        value = -1
    )
