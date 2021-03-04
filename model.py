import analysisFunctions as af
from pathlib import Path
import os
import matplotlib.pyplot as plt
import plotnine as gg
import seaborn as sns
import pandas as pd
import numpy as np


class Experiment:

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

    def __init__(self, media, osmolyte, temperature, date, folder, plot=False):
        self.name = media
        self.osmolyte = osmolyte
        self.temperature = temperature
        self.date = date
        self.folder = folder
        print(f"processing {self.name}")
        self.clean_data()
        self.combine_all_repeats()

        if plot == True:
            self.generate_plots()
            self.plot_gfp()

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
            temp_plate = Plate(media=self.name,
                               osmolyte=self.osmolyte,
                               temperature=self.temperature,
                               date=self.date,
                               folder=self.folder,
                               repeat_number=f"repeat_{num}",
                               data=analyzed_plate)
            temp_plate.calculate_max_growth_rate()
            temp_plate.subtract_wt()
            temp_plate.calculate_GFP_by_phase()
            list_of_repeats.append(temp_plate)
            self.list_of_repeats = list_of_repeats

    def generate_plots(self):
        for repeat in self.list_of_repeats:
            repeat.generate_plots()
            repeat.plot_growth_rate()


    def combine_all_repeats(self):
        all_dfs = []
        for repeat in self.list_of_repeats:
            repeat_name = repeat.repeat_number
            repeat.complete_df.loc[:, 'repeat'] = repeat_name
            all_dfs.append(repeat.complete_df)

        self.experiment_df = pd.concat(all_dfs).reset_index(drop=True)

    def plot_gfp(self):

        plot_path = os.path.join(self.folder, "Experiment_plots")
        Path(plot_path).mkdir(parents=True, exist_ok=True)
        split = self.experiment_df.groupby(['Group'])
        split_df = [split.get_group(x) for x in split.groups]

        for df in split_df:
            current_group = df['Group'].values[0]
            gfp_plot = (
                gg.ggplot(df) +
                gg.aes(x='Time', y='normalised_GFP/OD', color='variable') +
                gg.geom_point() +
                gg.ggtitle(current_group) +
                self.graph_theme
            )

            save_string = f"GFPOD_{current_group}.png"
            gg.ggsave(gfp_plot, os.path.join(
                plot_path, save_string), width=10, height=10, verbose = False)


        gfp_boxplot = sns.boxplot(x="osmolarity", y="normalised_GFP/OD", saturation=0.9, dodge=False, hue='phase', data=self.experiment_df)
        for patch in gfp_boxplot.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .6))

        figure = gfp_boxplot.get_figure()
        plot_path = os.path.join(self.folder, "Experiment_plots")

        save_string = f"GFP_boxplot_{self.name}.png"
        save_path = os.path.join(plot_path, save_string)

        figure.savefig(save_path, dpi=400)
        plt.close()


class Plate():

    def __init__(self, media, osmolyte, temperature, date, folder, repeat_number, data):

        self.name = media
        self.osmolyte = osmolyte
        self.temperature = temperature
        self.date = date
        self.folder = folder
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
                plot_path, save_string), width=10, height=10, verbose = False)

        for df in split_df:
            LN_group_plot = (
                gg.ggplot(df) +
                gg.aes(x='Time', y='log(OD)', color='variable') +
                gg.geom_point() +
                gg.ggtitle(df['Group'].values[0]) +
                self.graph_theme
            )
            save_string = f"lnOD_{df['Group'].values[0]}_{self.repeat_number}.png"
            gg.ggsave(LN_group_plot, os.path.join(plot_path, save_string), verbose = False)

    def calculate_max_growth_rate(self):

        filter_OD = 0.02
        length_exponential_phase = 8

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
                max_growth_rate = {'GrowthRate': 0, 'index_GrowthRate': np.nan}
                end_exponential_phase = {
                    'end_exponential': np.nan, 'index_end_exponential': np.nan}
                start_stationary_phase = {
                    'start_stationary': np.nan, 'index_start_stationary': np.nan}
            else:
                max_growth_rate = {'GrowthRate': new_df['GrowthRate'].max(
                ), 'index_GrowthRate': new_df['GrowthRate'].idxmax()}

                if max_growth_rate['index_GrowthRate']+length_exponential_phase < len(new_df):
                    end_exponential_phase = {'end_exponential': new_df['Time'][(
                        max_growth_rate['index_GrowthRate'] + length_exponential_phase)], 'index_end_exponential': (
                        max_growth_rate['index_GrowthRate'] + length_exponential_phase)}

                else:  # I added this else statement because in some cases runs were not done by the end of exponential phase or there is no clear phase
                    end_exponential_phase = {'end_exponential': new_df['Time'][(
                        max_growth_rate['index_GrowthRate'])], 'index_end_exponential': (
                        max_growth_rate['index_GrowthRate'])}

                
                start_stationary_phase_df = new_df[new_df['Time']
                                                   > end_exponential_phase['end_exponential']]
                index_start_stationary_phase = start_stationary_phase_df.index[
                    start_stationary_phase_df['GrowthRate'] < 0.05]
                
                if not index_start_stationary_phase.empty:
                    index_start_stationary_phase = index_start_stationary_phase[0]
                else:
                    index_start_stationary_phase = -1 #Here I set the index to the last measurement
                    #you could debate that this is incorrect as cells mayb not be in stationary phase yet
                
                if not index_start_stationary_phase == -1:
                    start_stationary_phase = {'start_stationary': start_stationary_phase_df['Time'][
                        index_start_stationary_phase], 'index_start_stationary': index_start_stationary_phase}
                if index_start_stationary_phase == -1:
                    start_stationary_phase = {'start_stationary': start_stationary_phase_df['Time'][
                        index_start_stationary_phase:], 'index_start_stationary': index_start_stationary_phase}

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
            df = df.dropna()
            try:
                df_mean = df['OD'].values[0:10].mean()
                df_std = df['OD'].values[0:10].std()
                std1 = df['Time'][df['OD']>(df_mean +df_std*1)].values[0]
                std5 = df['Time'][df['OD']>(df_mean + df_std*5)].values[0]
                std10 = df['Time'][df['OD']>(df_mean + df_std*10)].values[0]
                std20 = df['Time'][df['OD']>(df_mean + df_std*20)].values[0]

                current_variable = df['variable'].values[0]
                gr_plot = (
                    gg.ggplot(df) +
                    gg.aes(x='Time', y='GrowthRate', color='variable') +
                    gg.geom_point() +
                    gg.geom_hline(yintercept=self.max_growth_rate[current_variable]['GrowthRate'], color='black') +
                    gg.geom_vline(xintercept=std1, color='red') +
                    gg.geom_vline(xintercept=std5, color='red') +
                    gg.geom_vline(xintercept=std10, color='red') +
                    gg.geom_vline(xintercept=std20, color='red') +

                    gg.geom_vline(xintercept=self.end_exponential_phase[current_variable]['end_exponential'], color='blue') +
                    gg.geom_vline(xintercept=self.start_stationary_phase[current_variable]['start_stationary'], color='green') +
                    gg.ggtitle(current_variable) +
                    self.graph_theme
                )
                save_string = f"GR_{current_variable}_{self.repeat_number}.png"
                gg.ggsave(gr_plot, os.path.join(plot_path, save_string), verbose = False)
            except:
                current_variable = df['variable'].values[0]
                gr_plot = (
                    gg.ggplot(df) +
                    gg.aes(x='Time', y='GrowthRate', color='variable') +
                    gg.geom_point() +
                    gg.geom_hline(yintercept=self.max_growth_rate[current_variable]['GrowthRate'], color='black') +
                    gg.geom_vline(xintercept=std1, color='red') +
                    gg.geom_vline(xintercept=std5, color='red') +
                    gg.geom_vline(xintercept=std10, color='red') +

                    gg.geom_vline(xintercept=self.end_exponential_phase[current_variable]['end_exponential'], color='blue') +
                    gg.geom_vline(xintercept=self.start_stationary_phase[current_variable]['start_stationary'], color='green') +
                    gg.ggtitle(current_variable) +
                    self.graph_theme
                )
                save_string = f"GR_{current_variable}_{self.repeat_number}.png"
                gg.ggsave(gr_plot, os.path.join(plot_path, save_string), verbose = False)

        

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
                    mz_variable.loc[:, 'normalised_GFP/OD'] = subtract_col
                    mz_variable.loc[:, 'wt_OD'] = wt_variable['OD']
                    mz_variable.loc[:, 'wt_variable'] = wt_variable['variable']
                    mz_variable.loc[:, 'wt_GFP'] = wt_variable['GFP']
                    mz_variable.loc[:, 'wt_log(OD)'] = wt_variable['log(OD)']
                    mz_variable.loc[:, 'wt_GrowthRate'] = wt_variable['GrowthRate']
                    mz_variable.loc[:, 'wt_Group'] = wt_variable['Group']
                    subtracted_df.append(mz_variable)
                    self.normalized_df = pd.concat(
                        subtracted_df).dropna().reset_index(drop=True)

    def calculate_GFP_by_phase(self):
        # OK ive set it up so that I can calculate the max growth rate for everything
        # I can calculate GFP/OD which is in the normalized df
        # so now i need to calculate GFP by phase
        # exponential phase will be from timepoint of the max Growth rate to end of exponential
        # take all the datapoints and make a boxplot
        GFP_exponential_phase_dict = {}
        GFP_post_exponential_phase_dict = {}
        GFP_stationary_phase_dict = {}

        split = self.normalized_df.groupby('variable')
        split_df = [split.get_group(x).reset_index(drop=True)
                    for x in split.groups]

        for df in split_df:
            
            # calculate exponential phase df
            current_variable = df['variable'].values[0]

            location_max_growth = self.max_growth_rate[current_variable]['index_GrowthRate']
        # Now calculate + and - 4 of that location
            if np.isnan(location_max_growth):
                exponential_phase_approximation = np.nan
            else:
                exponential_phase_approximation = df.iloc[(
                    location_max_growth-4):(location_max_growth+4)]
            GFP_exponential_phase_dict[current_variable] = exponential_phase_approximation

            # calculate post exponential phase df
            # Here I take max growth + 5 because then when I join the dataframe together I am not dropping any data
            if np.isnan(location_max_growth):
                post_exponential_phase_approximation = np.nan
            else:
                location_end_exponential = location_max_growth+5
                location_start_stationary = self.start_stationary_phase[
                    current_variable]['index_start_stationary']
                post_exponential_phase_approximation = df.iloc[
                    location_end_exponential:location_start_stationary]
            GFP_post_exponential_phase_dict[current_variable] = post_exponential_phase_approximation

            # calculate stationary phase GFP
            if np.isnan(location_max_growth):
                stationary_phase_approximation = np.nan
            else:
                stationary_phase_approximation = df.iloc[location_start_stationary:]
            GFP_stationary_phase_dict[current_variable] = stationary_phase_approximation


        self.exponential_phase_df = pd.concat(
            pd.Series(GFP_exponential_phase_dict.values()).dropna().tolist()).reset_index(drop=True)
        self.exponential_phase_df.loc[:, 'phase'] = 'Exponential'
        self.post_exponential_phase_df = pd.concat(
            pd.Series(GFP_post_exponential_phase_dict.values()).dropna().tolist()).reset_index(drop=True)
        self.post_exponential_phase_df.loc[:, 'phase'] = 'Post-exponential'
        self.stationary_phase_df = pd.concat(
            pd.Series(GFP_stationary_phase_dict.values()).dropna().tolist()).reset_index(drop=True)
        self.stationary_phase_df.loc[:, 'phase'] = 'Stationary'
        self.complete_df = pd.concat(
            [self.exponential_phase_df, self.post_exponential_phase_df, self.stationary_phase_df]).reset_index(drop=True)
