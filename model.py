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
            axis_ticks=gg.element_line(colour="grey", size=0.3),
            panel_grid_major=gg.element_line(colour="grey", size=0.3),
            panel_grid_minor=gg.element_blank(),
            text=gg.element_text(size=21)
        )
    )

    def __init__(self, media, solute, temperature, date, folder, plot=False):
        self.name = media
        self.solute = solute
        self.temperature = temperature
        self.date = date
        self.folder = folder
        self.filter_value = 0.02
        print(f"processing {self.name}")
        self.clean_data()
        self.combine_all_repeats()

        if plot == True:
            self.generate_plots()
            self.plot_gfp()
            self.plot_max_values()

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
            analyzed_plate = af.analyze_plate(filepath, self.filter_value)
            temp_plate = Plate(media=self.name,
                               solute=self.solute,
                               temperature=self.temperature,
                               date=self.date,
                               folder=self.folder,
                               repeat_number=f"repeat_{num}",
                               data=analyzed_plate,
                               filter_value=self.filter_value)
            temp_plate.calculate_max_growth_rate()
            temp_plate.subtract_wt()
            temp_plate.calculate_gfp_by_phase_mz1()
            temp_plate.calculate_gfp_by_phase_wt()
            temp_plate.calculate_max_od_mz1()
            temp_plate.calculate_max_od_wt()
            temp_plate.add_max_values_to_df()
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

        gfp_df = self.experiment_df

        gfp_df = gfp_df[gfp_df['mz1_phase'] != 'Lag']
        split = gfp_df.groupby(['Group'])
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
                plot_path, save_string), width=10, height=10, verbose=False)

        gfp_boxplot = sns.boxplot(x="osmolarity", y="normalised_GFP/OD",
                                  saturation=0.9, dodge=False, hue='mz1_phase', data=gfp_df)
        for patch in gfp_boxplot.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .6))

        figure = gfp_boxplot.get_figure()
        plot_path = os.path.join(self.folder, "Experiment_plots")

        save_string = f"GFP_boxplot_{self.name}.png"
        save_path = os.path.join(plot_path, save_string)

        figure.savefig(save_path, dpi=400)
        plt.close()

    def plot_max_values(self):
        plot_path = os.path.join(self.folder, "Experiment_plots")
        Path(plot_path).mkdir(parents=True, exist_ok=True)
        for repeat in self.list_of_repeats:
            fig, (max_values_plot_gfp, max_values_plot_auc) = plt.subplots(nrows= 2, ncols=1,figsize = (11,7))
            
            sns.scatterplot(ax = max_values_plot_gfp,
                x="OD", y="GFP",
                data=repeat.max_values)        
            max_values_plot_gfp.set(title = repeat.repeat_number, ylabel = 'sfGFP (A.U.)', xlabel = 'OD[600]')

            sns.scatterplot(ax=max_values_plot_auc,
                x="OD", y="GFP_AUC",
                data=repeat.max_values)        
            max_values_plot_auc.set(title = f"{repeat.repeat_number} Area Under Curve", ylabel = 'sfGFP (AUC)', xlabel = 'OD[600]')

            save_string = f"Max_values_{repeat.repeat_number}_{self.name}.png"
            save_path = os.path.join(plot_path, save_string)

            plt.tight_layout()

            fig.savefig(save_path, dpi=400)
            plt.close()


class Plate():

    def __init__(self, media, solute, temperature, date, folder, repeat_number, data, filter_value):

        self.name = media
        self.solute = solute
        self.temperature = temperature
        self.date = date
        self.folder = folder
        self.repeat_number = repeat_number
        self.data = data
        self.filter_value = filter_value

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
            od_group_plot = (
                gg.ggplot(df) +
                gg.aes(x='Time', y='OD', color='variable') +
                gg.geom_point() +
                gg.ggtitle(current_group) +
                self.graph_theme
            )

            save_string = f"OD_{current_group}_{self.repeat_number}.png"
            gg.ggsave(od_group_plot, os.path.join(
                plot_path, save_string), width=10, height=10, verbose=False)

        for df in split_df:
            LN_group_plot = (
                gg.ggplot(df) +
                gg.aes(x='Time', y='log(OD)', color='variable') +
                gg.geom_point() +
                gg.ggtitle(df['Group'].values[0]) +
                self.graph_theme
            )
            save_string = f"lnOD_{df['Group'].values[0]}_{self.repeat_number}.png"
            gg.ggsave(LN_group_plot, os.path.join(
                plot_path, save_string), verbose=False)

    def calculate_max_growth_rate(self):

        filter_OD = self.filter_value
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
                    start_stationary_phase_df['GrowthRate'] < self.filter_value]

                if not index_start_stationary_phase.empty:
                    index_start_stationary_phase = index_start_stationary_phase[0]
                else:
                    index_start_stationary_phase = -1  # Here I set the index to the last measurement
                    # you could debate that this is incorrect as cells mayb not be in stationary phase yet

                if not index_start_stationary_phase == -1:
                    start_stationary_phase = {'start_stationary': start_stationary_phase_df['Time'][
                        index_start_stationary_phase], 'index_start_stationary': index_start_stationary_phase}
                if index_start_stationary_phase == -1:
                    start_stationary_phase = {'start_stationary': start_stationary_phase_df['Time'][
                        index_start_stationary_phase:].values[0], 'index_start_stationary': index_start_stationary_phase}

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
                std1 = df['Time'][df['OD'] > (df_mean + df_std*1)].values[0]
                std3 = df['Time'][df['OD'] > (df_mean + df_std*3)].values[0]
                std5 = df['Time'][df['OD'] > (df_mean + df_std*5)].values[0]
                std10 = df['Time'][df['OD'] > (df_mean + df_std*10)].values[0]

                current_variable = df['variable'].values[0]
                gr_plot = (
                    gg.ggplot(df) +
                    gg.aes(x='Time', y='GrowthRate', color='variable') +
                    gg.geom_point() +
                    gg.geom_hline(yintercept=self.max_growth_rate[current_variable]['GrowthRate'], color='black') +
                    gg.geom_vline(xintercept=std1, color='red') +
                    gg.geom_vline(xintercept=std3, color='red') +
                    gg.geom_vline(xintercept=std5, color='red') +
                    gg.geom_vline(xintercept=std10, color='red') +

                    gg.geom_vline(xintercept=self.end_exponential_phase[current_variable]['end_exponential'], color='blue') +
                    gg.geom_vline(xintercept=self.start_stationary_phase[current_variable]['start_stationary'], color='green') +
                    gg.ggtitle(current_variable) +
                    gg.xlim(0, 20) +
                    self.graph_theme
                )
                save_string = f"GR_{current_variable}_{self.repeat_number}.png"
                gg.ggsave(gr_plot, os.path.join(
                    plot_path, save_string), verbose=False)
            except:
                print("error not possible to create plot")

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
                    mz_variable.loc[:,
                                    'wt_GrowthRate'] = wt_variable['GrowthRate']
                    mz_variable.loc[:, 'wt_Group'] = wt_variable['Group']
                    subtracted_df.append(mz_variable)
                    self.normalized_df = pd.concat(
                        subtracted_df).dropna().reset_index(drop=True)

    def calculate_gfp_by_phase_mz1(self):
        # OK ive set it up so that I can calculate the max growth rate for everything
        # I can calculate GFP/OD which is in the normalized df
        # so now i need to calculate GFP by phase
        # exponential phase will be from timepoint of the max Growth rate to end of exponential
        # take all the datapoints and make a boxplot
        gfp_lag_phase_dict = {}
        gfp_exponential_phase_dict = {}
        gfp_post_exponential_phase_dict = {}
        gfp_stationary_phase_dict = {}

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
                lag_phase_estimation = df
            else:
                exponential_phase_approximation = df.iloc[(
                    location_max_growth-4):(location_max_growth+4)]
                lag_phase_estimation = df.iloc[:location_max_growth-4]
            gfp_exponential_phase_dict[current_variable] = exponential_phase_approximation
            gfp_lag_phase_dict[current_variable] = lag_phase_estimation

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
            gfp_post_exponential_phase_dict[current_variable] = post_exponential_phase_approximation

            # calculate stationary phase GFP
            if np.isnan(location_max_growth):
                stationary_phase_approximation = np.nan
            else:
                stationary_phase_approximation = df.iloc[location_start_stationary:]
            gfp_stationary_phase_dict[current_variable] = stationary_phase_approximation

        self.lag_phase_df = pd.concat(
            pd.Series(gfp_lag_phase_dict.values()).dropna().tolist()).reset_index(drop=True)
        self.lag_phase_df.loc[:, 'mz1_phase'] = 'Lag'
        self.exponential_phase_df = pd.concat(
            pd.Series(gfp_exponential_phase_dict.values()).dropna().tolist()).reset_index(drop=True)
        self.exponential_phase_df.loc[:, 'mz1_phase'] = 'Exponential'
        self.post_exponential_phase_df = pd.concat(
            pd.Series(gfp_post_exponential_phase_dict.values()).dropna().tolist()).reset_index(drop=True)
        self.post_exponential_phase_df.loc[:, 'mz1_phase'] = 'Post-exponential'
        self.stationary_phase_df = pd.concat(
            pd.Series(gfp_stationary_phase_dict.values()).dropna().tolist()).reset_index(drop=True)
        self.stationary_phase_df.loc[:, 'mz1_phase'] = 'Stationary'
        self.normalized_df_pre_wt_phase = pd.concat(
            [self.lag_phase_df, self.exponential_phase_df, self.post_exponential_phase_df, self.stationary_phase_df]).reset_index(drop=True)

    def calculate_gfp_by_phase_wt(self):
        # OK ive set it up so that I can calculate the max growth rate for everything
        # I can calculate GFP/OD which is in the normalized df
        # so now i need to calculate GFP by phase
        # exponential phase will be from timepoint of the max Growth rate to end of exponential
        # take all the datapoints and make a boxplot
        wt_lag_phase_dict = {}
        wt_exponential_phase_dict = {}
        wt_post_exponential_phase_dict = {}
        wt_stationary_phase_dict = {}

        split = self.normalized_df_pre_wt_phase.groupby('wt_variable')
        split_df = [split.get_group(x).reset_index(drop=True)
                    for x in split.groups]

        for df in split_df:
            # calculate exponential phase df
            current_variable = df['wt_variable'].values[0]
            location_max_growth = self.max_growth_rate[current_variable]['index_GrowthRate']
        # Now calculate + and - 4 of that location
            if np.isnan(location_max_growth):
                exponential_phase_approximation = np.nan
                lag_phase_estimation = df
            else:
                exponential_phase_approximation = df.iloc[(
                    location_max_growth-4):(location_max_growth+4)]
                lag_phase_estimation = df.iloc[:location_max_growth-4]

            wt_exponential_phase_dict[current_variable] = exponential_phase_approximation
            wt_lag_phase_dict[current_variable] = lag_phase_estimation


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
            wt_post_exponential_phase_dict[current_variable] = post_exponential_phase_approximation

            # calculate stationary phase wt
            if np.isnan(location_max_growth):
                stationary_phase_approximation = np.nan
            else:
                stationary_phase_approximation = df.iloc[location_start_stationary:]
            wt_stationary_phase_dict[current_variable] = stationary_phase_approximation

        self.lag_phase_df = pd.concat(
            pd.Series(wt_lag_phase_dict.values()).dropna().tolist()).reset_index(drop=True)
        self.lag_phase_df.loc[:, 'wt_phase'] = 'Lag'
        self.exponential_phase_df_wt = pd.concat(
            pd.Series(wt_exponential_phase_dict.values()).dropna().tolist()).reset_index(drop=True)
        self.exponential_phase_df_wt.loc[:, 'wt_phase'] = 'Exponential'
        self.post_exponential_phase_df_wt = pd.concat(
            pd.Series(wt_post_exponential_phase_dict.values()).dropna().tolist()).reset_index(drop=True)
        self.post_exponential_phase_df_wt.loc[:, 'wt_phase'] = 'Post-exponential'
        self.stationary_phase_df_wt = pd.concat(
            pd.Series(wt_stationary_phase_dict.values()).dropna().tolist()).reset_index(drop=True)
        self.stationary_phase_df_wt.loc[:, 'wt_phase'] = 'Stationary'
        self.complete_df = pd.concat(
            [self.lag_phase_df, self.exponential_phase_df_wt, self.post_exponential_phase_df_wt, self.stationary_phase_df_wt]).reset_index(drop=True)

    def calculate_max_od_mz1(self):

        split = self.complete_df.groupby('variable')
        split_df = [split.get_group(x).reset_index(drop=True)
                    for x in split.groups]

        self.max_od_mz1 = {}
        self.max_gfp = {}
        self.gfp_area_under_curve = {}

        for df in split_df:
            current_variable = df['variable'].values[0]
            start_stationary_index = self.start_stationary_phase[current_variable]['start_stationary']
            
            try:
                od_start_stationary = df.loc[df['Time'] >=
                                            start_stationary_index, 'OD'].values[0]
                
            except:
                '''
                Here I catch the error that there is no start of the stationary phase.
                This means that the OD will eb undervalued.
                '''
                od_start_stationary = df.iloc[-1]['OD']

            df_gfp_exponential = df[(df['mz1_phase']=='Lag') | (df['mz1_phase']=='exponential') | (df['OD'] > self.filter_value)]
            gfp_max_value = df_gfp_exponential['normalised_GFP/OD'].mean()
            gfp_area_under_curve = np.trapz(df_gfp_exponential['normalised_GFP/OD'], df_gfp_exponential['Time'], dx=1.0, axis=-1)
            
            self.max_od_mz1[current_variable] = od_start_stationary
            self.max_gfp[current_variable] = gfp_max_value
            self.gfp_area_under_curve[current_variable] = gfp_area_under_curve

        merged_dict = [self.max_od_mz1, self.max_gfp, self.gfp_area_under_curve]
        self.max_values = {}
        for k in self.max_od_mz1.keys():
            self.max_values[k] = tuple(d[k] for d in merged_dict)

        self.max_values = pd.DataFrame(self.max_values).transpose()
        self.max_values.columns = ['OD', 'GFP', 'GFP_AUC']
        self.max_values['repeat'] = self.repeat_number

    def calculate_max_od_wt(self):

        split = self.complete_df.groupby('wt_variable')
        split_df = [split.get_group(x).reset_index(drop=True)
                    for x in split.groups]

        self.max_od_wt = {}
        self.max_gfp_wt = {}
        self.gfp_area_under_curve_wt = {}

        for df in split_df:
            current_variable = df['wt_variable'].values[0]
            start_stationary_index = self.start_stationary_phase[current_variable]['start_stationary']
            
            try:
                od_start_stationary = df.loc[df['Time'] >=
                                            start_stationary_index, 'OD'].values[0]
            except:
                '''
                Here I catch the error that there is no start of the stationary phase.
                This means that the OD will eb undervalued.
                '''
                od_start_stationary = df.iloc[-1]['OD']

            gfp_max_value = df['normalised_GFP/OD'].max()
            gfp_area_under_curve_wt = np.trapz(df['normalised_GFP/OD'], df['Time'], dx=1.0, axis=-1)

            self.max_od_wt[current_variable] = od_start_stationary
            self.max_gfp_wt[current_variable] = gfp_max_value
            self.gfp_area_under_curve_wt[current_variable] = gfp_area_under_curve_wt

        merged_dict = [self.max_od_wt, self.max_gfp_wt, self.gfp_area_under_curve_wt]
        self.max_values = {}
        for k in self.max_od_wt.keys():
            self.max_values[k] = tuple(d[k] for d in merged_dict)

        self.max_values = pd.DataFrame(self.max_values).transpose()
        self.max_values.columns = ['OD', 'GFP', 'GFP_AUC']
        self.max_values['repeat'] = self.repeat_number
    
    def add_max_values_to_df(self):
        '''
        Here I add all the max values to the dataframe to make it easier to plot later on
        '''
        # Here is the max growth rate
        self.complete_df['MZ1_max_growth_rate'] = self.complete_df['variable']
        self.complete_df['MZ1_max_growth_rate'] = self.complete_df['MZ1_max_growth_rate'].map(self.max_growth_rate)
        self.complete_df['MZ1_max_growth_rate'] = [d.get('GrowthRate') for d in self.complete_df.MZ1_max_growth_rate]

        self.complete_df['wt_max_growth_rate'] = self.complete_df['wt_variable']
        self.complete_df['wt_max_growth_rate'] = self.complete_df['wt_max_growth_rate'].map(self.max_growth_rate)
        self.complete_df['wt_max_growth_rate'] = [d.get('GrowthRate') for d in self.complete_df.wt_max_growth_rate]

        # here is the max GFP This is the normalised GFP and thus has values for MZ1 only
        self.complete_df['MZ1_max_gfp'] = self.complete_df['variable']
        self.complete_df['MZ1_max_gfp'] = self.complete_df['MZ1_max_gfp'].map(self.max_gfp)

        self.complete_df['MZ1_gfp_AUC'] = self.complete_df['variable']
        self.complete_df['MZ1_gfp_AUC'] = self.complete_df['MZ1_gfp_AUC'].map(self.gfp_area_under_curve)

        # here is the max OD
        self.complete_df['MZ1_max_od'] = self.complete_df['variable']
        self.complete_df['MZ1_max_od'] = self.complete_df['MZ1_max_od'].map(self.max_od_mz1)
        self.complete_df['wt_max_od'] = self.complete_df['wt_variable']
        self.complete_df['wt_max_od'] = self.complete_df['wt_max_od'].map(self.max_od_wt)
