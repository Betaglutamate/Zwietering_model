import analysisFunctions as af
from pathlib import Path
import os
import plotnine as gg
import pandas as pd
import numpy as np


class Experiment:

    def __init__(self, media, osmolyte, temperature, date, folder):
        self.name = media
        self.osmolyte = osmolyte
        self.temperature = temperature
        self.date = date
        self.folder = folder

    def clean_data(self):
        files_to_analyze = []
        for root, dirs, files in os.walk(self.folder):
            for filename in files:
                if filename.endswith(".xlsx"):
                    files_to_analyze.append(
                        {"root": root, "filename": filename})

        list_of_repeats = []

        for num, repeat in enumerate(files_to_analyze):
            filepath = os.path.join(repeat['root'], repeat['filename'])
            analyzed_plate = af.analyze_plate(filepath)
            temp_plate = Plate(self.name, self.osmolyte, self.temperature,
                               self.date, self.folder, f"repeat_{num}", analyzed_plate)
            temp_plate.calculate_max_growth_rate()
            temp_plate.subtract_wt()
            list_of_repeats.append(temp_plate)
            self.list_of_repeats = list_of_repeats


class Plate(Experiment):

    def __init__(self, media, osmolyte, temperature, date, folder, repeat_number, data):
        super().__init__(media, osmolyte, temperature, date, folder)
        self.repeat_number = repeat_number
        self.data = data

    # We use theme_bw so the changes are consistent in all plots

    gg.theme_set(gg.theme_bw())
    graph_theme = (
        gg.theme(
            plot_title=gg.element_text(face="bold", size=12),
            legend_background=gg.element_rect(
                fill="white", size=4, colour="white"),
            # legend.justification = c(0, 1),
            # legend.position = c(0, 1),
            axis_ticks=gg.element_line(colour="grey", size=0.3),
            panel_grid_major=gg.element_line(colour="grey", size=0.3),
            panel_grid_minor=gg.element_blank(),
            text=gg.element_text(size=21)
        )
    )

    def generate_plots(self):
        """
        This function takes in a long Df it should then output a graph for every
        group
        """
        plot_path = os.path.join(self.folder, "plots", self.repeat_number)
        Path(plot_path).mkdir(parents=True, exist_ok=True)
        split = self.data.groupby('Group')
        split_df = [split.get_group(x) for x in split.groups]

        for df in split_df:
            current_group = df['Group'].values[0]
            OD_group_plot = (
                gg.ggplot(df) +
                gg.aes(x='Time', y='OD', color='variable') +
                gg.geom_point() +
                gg.ggtitle(current_group) +
                self.graph_theme
            )

            save_string = f"OD_{current_group}_{self.repeat_number}.png"
            gg.ggsave(OD_group_plot, os.path.join(
                plot_path, save_string), width=10, height=10)

        for df in split_df:
            LN_group_plot = (
                gg.ggplot(df) +
                gg.aes(x='Time', y='log(OD)', color='variable') +
                gg.geom_point() +
                gg.ggtitle(df['Group'].values[0]) +
                self.graph_theme
            )
            save_string = f"lnOD_{df['Group'].values[0]}_{self.repeat_number}.png"
            gg.ggsave(LN_group_plot, os.path.join(plot_path, save_string))

    def calculate_max_growth_rate(self):

        filter_OD = 0.02

        split = self.data.groupby('variable')
        split_df = [split.get_group(x) for x in split.groups]

        max_growth_rate_dict = {}
        end_exponential_phase_dict = {}
        start_stationary_phase_dict = {}

        for temp_df in split_df:
            temp_df = temp_df.reset_index(drop=True)
            current_variable = temp_df['variable'].values[0]
            new_df = temp_df[temp_df['OD'] > filter_OD]
            if new_df.empty:
                max_growth_rate = 0
                end_exponential_phase = np.nan
                start_stationary_phase = np.nan
            else:
                max_growth_rate = {'GrowthRate': new_df['GrowthRate'].max(), 'index_GrowthRate': new_df['GrowthRate'].idxmax()}
                end_exponential_phase = {'end_exponential': new_df['Time'][(
                    max_growth_rate['index_GrowthRate'] + 8)], 'index_end_exponential': (
                    max_growth_rate['index_GrowthRate'] + 8)}

                start_stationary_phase_df = new_df[new_df['Time']
                                                   > end_exponential_phase['end_exponential']]
                index_start_stationary_phase = start_stationary_phase_df.index[
                    start_stationary_phase_df['GrowthRate'] < 0.05][0]
                start_stationary_phase = {'start_stationary': start_stationary_phase_df['Time'][index_start_stationary_phase], 'index_start_stationary': index_start_stationary_phase} 

            max_growth_rate_dict[current_variable] = max_growth_rate
            end_exponential_phase_dict[current_variable] = end_exponential_phase
            start_stationary_phase_dict[current_variable] = start_stationary_phase

        self.max_growth_rate = max_growth_rate_dict
        self.end_exponential_phase = end_exponential_phase_dict
        self.start_stationary_phase = start_stationary_phase_dict

    def plot_growth_rate(self):

        plot_path = os.path.join(self.folder, "plots",
                                 self.repeat_number, "GrowthRate")
        Path(plot_path).mkdir(parents=True, exist_ok=True)

        split = self.data.groupby('variable')
        split_df = [split.get_group(x) for x in split.groups]

        for df in split_df:
            current_variable = df['variable'].values[0]
            LN_group_plot = (
                gg.ggplot(df) +
                gg.aes(x='Time', y='GrowthRate', color='variable') +
                gg.geom_point() +
                gg.geom_hline(yintercept=self.max_growth_rate[current_variable]['GrowthRate'], color='black') +
                gg.geom_vline(xintercept = self.end_exponential_phase[current_variable]['end_exponential'], color='blue') +
                gg.geom_vline(xintercept = self.start_stationary_phase[current_variable]['start_stationary'], color='green') +
                gg.ggtitle(current_variable) +
                self.graph_theme
            )
            save_string = f"GR_{current_variable}_{self.repeat_number}.png"
            gg.ggsave(LN_group_plot, os.path.join(plot_path, save_string))

    def subtract_wt(self):
        # Now I should split self.data into containing MZ and WT
        wt_df = self.data[self.data['variable'].str.upper().str.match('WT')]
        mz_df = self.data[self.data['variable'].str.upper().str.match('MZ')]

        # Then I need to match the last 6 chars of the variable and subtract
        split = wt_df.groupby('variable')
        split_wt = [split.get_group(x) for x in split.groups]

        split = mz_df.groupby('variable')
        split_mz = [split.get_group(x) for x in split.groups]

        subtracted_df = []

        for wt_variable in split_wt:
            for mz_variable in split_mz:
                if mz_variable['variable'].values[0][-6:] == wt_variable['variable'].values[0][-6:]:
                    mz_variable = mz_variable.reset_index(drop=True)
                    wt_variable = wt_variable.reset_index(drop=True)
                    subtract_col = mz_variable['GFP/OD'].reset_index(
                        drop=True) - wt_variable['GFP/OD'].reset_index(drop=True)
                    mz_variable['normalised_GFP/OD'] = subtract_col
                    mz_variable['wt_variable'] = wt_variable['variable']
                    mz_variable['wt_OD'] = wt_variable['OD']
                    mz_variable['wt_GFP'] = wt_variable['GFP']
                    mz_variable['wt_log(OD)'] = wt_variable['log(OD)']
                    mz_variable['wt_GrowthRate'] = wt_variable['GrowthRate']
                    mz_variable['wt_Group'] = wt_variable['Group']
                    subtracted_df.append(mz_variable)
                    self.normalized_df = pd.concat(
                        subtracted_df).dropna().reset_index(drop=True)

    def calculate_GFP_by_phase(self):
        #OK ive set it up so that I can calculate the max growth rate for everything
        # I can calculate GFP/OD which is in the normalized df
        # so now i need to calculate GFP by phase
        # exponential phase will be from timepoint of the max Growth rate to end of exponential
        # take all the datapoints and make a boxplot
        GFP_exponential_phase_dict = {}
        
        split = self.normalized_df.groupby('variable')
        split_df = [split.get_group(x).reset_index(drop=True) for x in split.groups]

        for df in split_df:

            current_variable = df['variable'].values[0]
            location_max_growth = self.max_growth_rate[current_variable]['index_GrowthRate']
            #Now calculate + and - 4 of that location
            exponential_phase_approximation = df.iloc[(location_max_growth-4):(location_max_growth+4)]
            mean_exponential_phase_approximation = exponential_phase_approximation ["GFP/OD"].mean()
            GFP_exponential_phase_dict[current_variable] = mean_exponential_phase_approximation

        self.GFP_exponential_phase = pd.DataFrame(test.items(), columns=['variable', 'meanGFP'])






