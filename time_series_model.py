import run_plate as af
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotnine as gg
from scipy.stats import linregress
import json
import pickle



class Experiment:

    def __init__(self, media, solute, temperature, date, folder, plot=False, label='standard'):
        self.name = media
        self.solute = solute
        self.temperature = temperature
        self.date = date
        self.folder = folder
        self.filter_value = 0.01
        self.length_exponential_phase = 8
        self.plot_bool = plot
        self.label = label
        print(f"processing {self.name}")
        self.clean_data()

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
            analyzed_plate = af.analyze_plate(filepath, self.filter_value, self.label)
            temp_plate = Plate(media=self.name,
                               solute=self.solute,
                               temperature=self.temperature,
                               date=self.date,
                               folder=self.folder,
                               repeat_number=f"repeat_{num}",
                               data=analyzed_plate,
                               filter_value=self.filter_value,
                               plot_bool=self.plot_bool)
            temp_plate.fit_df()
            temp_plate.calculate_max_growth_rate()
            # temp_plate.calculate_growth_phase()

            if self.plot_bool:
                sns.set_style("whitegrid")
                temp_plate.visualize_growth_rate()

            temp_plate.align_data()
            temp_plate.subtract_wt()
            temp_plate.calculate_max_gfp()
            list_of_repeats.append(temp_plate)
            self.list_of_repeats = list_of_repeats

        # combine the repeats outside of loop
        self.combine_all_repeats()

        if self.plot_bool:
            self.plot_experiment()

    def combine_all_repeats(self):
        all_dfs = []
        all_data = []
        for repeat in self.list_of_repeats:
            repeat_name = repeat.repeat_number
            repeat.final_df.loc[:, 'repeat'] = repeat_name
            repeat.data.loc[:, 'repeat'] = repeat_name
            repeat.data.loc[:, 'experiment'] = (f"{self.name}_{self.date}")
            all_dfs.append(repeat.final_df)
            all_data.append(repeat.data)

        self.full_data = pd.concat(all_data).reset_index(drop=True)
        self.experiment_df = pd.concat(all_dfs).reset_index(drop=True)


    def plot_experiment(self):

        plot_path = os.path.join(self.folder, "Experiment_plots")
        Path(plot_path).mkdir(parents=True, exist_ok=True)

        gfp_df = self.experiment_df

        gfp_boxplot = sns.boxplot(x="mz1_osmolarity", y="normalised_GFP/OD",
                                  saturation=0.9, dodge=False, hue='mz1_growth_phase', data=gfp_df)
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

    def __init__(self,
                 media,
                 solute,
                 temperature,
                 date,
                 folder,
                 repeat_number,
                 data,
                 filter_value,
                 plot_bool):

        self.name = media
        self.solute = solute
        self.temperature = temperature
        self.date = date
        self.folder = folder
        self.repeat_number = repeat_number
        self.data = data
        self.filter_value = filter_value
        self.plot = plot_bool

    def fit_df(self):

        #I need to load the model here
        with open('time_series/rocket.pkl', 'rb') as input:
            list_classifier = pickle.load(input)

        rocket = list_classifier[0]
        classifier = list_classifier[1]
        length_window=20
        
        self.data = self.calculate_growth_rate(self.data)

        growth_phase_df = []

        for name, df in self.data.groupby('variable'):
            df = df.copy().reset_index(drop=True)
            full_df = self.generate_fitted_plots(df, classifier, rocket, self.plot)
            growth_phase_df.append(full_df)
        
        self.growth_phase_df = pd.concat(growth_phase_df)

    def align_data(self):
        self.aligned_data = af.align_df(self.params_df, self.filter_value)


    def calculate_growth_rate(self, df):
        window_size = 8

        annotated_df = []
        df = df.copy()

        for name, df in df.groupby('variable'):
            regress_list = []
            for length in range(len(df)):
                res = linregress(
                    x=df['Time'].values[length:length+window_size],
                    y=df['log(OD)'].values[length:length+window_size]
                )
                regress_list.append(res.slope)

            df['growth_rate'] = pd.Series(regress_list, index=df.index)
            annotated_df.append(df)

        growth_rate_calculated_df = pd.concat(
            annotated_df).reset_index(drop=True)

        return growth_rate_calculated_df
    
    def calculate_max_growth_rate(self):

        df = self.growth_phase_df.copy()
        added_params_df = []

        for name, df in df.groupby('variable'):
            df_exponential = df[df['growth_phase'] ==  'exponential_phase']
            max_growth_rate = np.max(df_exponential['growth_rate'])
            max_yield = np.max(df['OD'])
            df['max_growth_rate'] = max_growth_rate
            df['max_yield'] = max_yield
            added_params_df.append(df)
        
        params_df = pd.concat(added_params_df)

        self.params_df = params_df

    def visualize_growth_rate(self):

        for name, df in self.growth_phase_df.groupby('Group'):
            fig, [ax1, ax2] = plt.subplots(2)
            sns.scatterplot(data=df, x='Time', y='log(OD)',
                            hue='growth_phase', ax=ax1)
            plot_path = os.path.join(
                self.folder, "Experiment_plots", "growth_phase")
            Path(plot_path).mkdir(parents=True, exist_ok=True)
            ax1.set_title(name)
            ax1.set_xlim(0, 80)
            sns.scatterplot(data=df, x='Time', y='growth_rate',
                            hue='growth_phase', ax=ax2)
            ax2.set_xlim(0, 80)
            plt.tight_layout()
            plt.savefig(
                f"{os.path.join(plot_path, name)}_growth_rate_{self.repeat_number}.png")
            plt.close()


        # for name, df in self.data.groupby('Group'):
        #     fig, ax = plt.subplots()
        #     sns.scatterplot(data=df, x='OD', y='growth_rate',
        #                     hue='growth_phase', ax=ax)
        #     plot_path = os.path.join(
        #         self.folder, "Experiment_plots", "growth_phase")
        #     Path(plot_path).mkdir(parents=True, exist_ok=True)
        #     ax.set_title(name)
        #     plt.savefig(
        #         f"{os.path.join(plot_path, name)}_growth_rate_vs_OD_{self.repeat_number}.png")
        #     plt.close()

# now that I have the data I need I will align the dataframes then split them up
    def split_data_frames(self, df):
        df_mz = df[df['Group'].str[0:2] == 'MZ'].reset_index(
            drop=True).add_prefix("mz1_")
        # I am dropping these because they are identical between dataframes.
        cols_to_drop = ['Time', 'osmolarity']
        df_wt = df[df['Group'].str[0:2] == 'WT'].drop(
            columns=cols_to_drop).reset_index(drop=True).add_prefix("wt_")
        return df_wt, df_mz

    def subtract_wt(self):
        # Now I should split self.data into containing MZ and WT
        wt_df, mz_df = self.split_data_frames(self.aligned_data)

        subtracted_df = []
        for wt_name, wt_df_loop in wt_df.groupby('wt_variable'):
            for mz_name, mz_df_loop in mz_df.groupby('mz1_variable'):
                if mz_name[-6:] == wt_name[-6:]:
                    mz_df_loop = mz_df_loop.reset_index(drop=True).copy()
                    wt_df_loop = wt_df_loop.reset_index(drop=True).copy()
                    subtract_col = mz_df_loop['mz1_GFP/OD'] - \
                        wt_df_loop['wt_GFP/OD']
                    mz_df_loop['normalised_GFP/OD'] = subtract_col
                    mz_df_loop['wt_OD'] = wt_df_loop['wt_OD']
                    mz_df_loop['wt_log(OD)'] = wt_df_loop['wt_log(OD)']
                    mz_df_loop['wt_growth_rate'] = wt_df_loop['wt_growth_rate']
                    mz_df_loop['wt_max_growth_rate'] = wt_df_loop['wt_max_growth_rate']
                    mz_df_loop['wt_max_yield'] = wt_df_loop['wt_max_yield']
                    subtracted_df.append(mz_df_loop)
        self.normalized_df = pd.concat(
            subtracted_df).dropna().reset_index(drop=True)

    def calculate_max_gfp(self):
        full_df = self.normalized_df.copy()

        max_gfp_df = []
        for name, df in full_df.groupby('mz1_variable'):
            df_gfp_exponential = df[(
                df['mz1_growth_phase'] == 'exponential_phase')]
            df['max_gfp'] = df_gfp_exponential['normalised_GFP/OD'].mean()
            df['auc_gfp'] = np.trapz(
                df_gfp_exponential['normalised_GFP/OD'], df_gfp_exponential['mz1_Time'], dx=1.0, axis=-1)

            max_gfp_df.append(df)

        self.final_df = pd.concat(max_gfp_df)

    def create_subcurve(self, single_variable, length_window):
        X_values = []
        time_values = []
        y_growth_values = []
        for start_window in (range(len(single_variable.index)-length_window)):
            window = single_variable.iloc[start_window: start_window+length_window]
            X = window['OD']
            y_growth = window['growth_rate']
            time = window['Time'].values[0]
            X_values.append(X)
            time_values.append(time)
            y_growth_values.append(y_growth)

        return X_values, y_growth_values, time_values


    def generate_fitted_plots(self, data, classifier, rocket, plot_bool, length_window=20):
        
        save_path = os.path.join(self.folder, 'rocket_plots')
        Path(save_path).mkdir(parents=True, exist_ok=True)

        predicted_df = []

        for name, variable in data.groupby('variable'):
            X, y_growth, time = self.create_subcurve(variable,length_window)
        
            xTest = X
            yGrowth = y_growth
            xTime = time

            df_test_x = pd.DataFrame({"dim_0": xTest, "dim_1": yGrowth})

            try:
                X_test_transform = rocket.transform(df_test_x)
                #make values for plotting
                transformed = classifier.predict(X_test_transform)
            except:
                transformed = "error"

            xOD = np.fromiter((x.values[0] for x in xTest), float)
            xGR = np.fromiter((x.values[0] for x in yGrowth), float)

            rebuilt_df = variable.iloc[0:-20]
            rebuilt_df['growth_phase'] = transformed
            predicted_df.append(rebuilt_df)

            if plot_bool:
                try:
                    fig, [ax1, ax2] = plt.subplots(2)
                    sns.scatterplot(x=xTime, y=np.log(xOD), hue=transformed, ax=ax1, s=5)
                    sns.scatterplot(x=xTime, y=xGR, hue=transformed, ax=ax2, s=5)
                    ax2.set(title='Growth Rate')
                    ax1.set(title='ln(OD)')
                    ax2.get_legend().remove()
                    plt.suptitle(name)

                    plt.tight_layout()
                    plt.savefig(f'{os.path.join(save_path, name)}.png', transparent = False, dpi=300)
                    plt.close()
                except ValueError:
                    print(f"could not create plot {name}")
        
        full_df = pd.concat(predicted_df)
        
        return full_df
        
        

    
