import run_plate as af
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import linregress


class Experiment:

    def __init__(self, media, solute, temperature, date, folder, plot=False):
        self.name = media
        self.solute = solute
        self.temperature = temperature
        self.date = date
        self.folder = folder
        self.filter_value = 0.01
        self.length_exponential_phase = 8
        self.plot_bool=plot
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
            analyzed_plate = af.analyze_plate(filepath, self.filter_value)
            temp_plate = Plate(media=self.name,
                               solute=self.solute,
                               temperature=self.temperature,
                               date=self.date,
                               folder=self.folder,
                               repeat_number=f"repeat_{num}",
                               data=analyzed_plate,
                               filter_value=self.filter_value,
                               plot_bool= self.plot_bool)
            temp_plate.fit_df()
            temp_plate.calculate_growth_phase()

            if self.plot_bool:
                temp_plate.visualize_growth_rate()

            temp_plate.align_data()
            temp_plate.subtract_wt()
            

            
            

            list_of_repeats.append(temp_plate)
            self.list_of_repeats = list_of_repeats


class Plate():

    def __init__(self, media, solute, temperature, date, folder, repeat_number, data, filter_value, plot_bool):

        self.name = media
        self.solute = solute
        self.temperature = temperature
        self.date = date
        self.folder = folder
        self.repeat_number = repeat_number
        self.data = data
        self.filter_value = filter_value
        self.plot = plot_bool

    def zwietering_model(self, t, A, Kz, Tlag):
        return  A*np.exp(-np.exp(((np.e*Kz)/A)*(Tlag-t)+1))
        
    def fit_df(self):
        
        names = []
        gr = []
        my = []
        lt = []
        
        fit_df = []

        for name, df in self.data.groupby('variable'):

            df = df.copy()

            df['zwieter'] = (np.log(df['OD'] / (df['OD'].values[0])))
            xData = df['Time']
            yData = df['zwieter']
            p0_test = [1, 0.8, 3]

            popt, _ = curve_fit(f=self.zwietering_model, xdata=xData, ydata=yData, p0=p0_test, maxfev = 1000)

            if self.plot:
                self.plot_fitted_df(xData, yData, popt, name)

            growth_rate = popt[1]
            max_yield = popt[0]
            lag_time = popt[2]
            max_yield = np.exp(max_yield)*df['OD'].values[0]

            names.append(name)
            lt.append(lag_time)
            my.append(max_yield)
            gr.append(growth_rate)
            new_df = pd.DataFrame({"name": names, "gr":gr, "my":my, "lt": lt})
            self.new_df = new_df

            df['lag_time'] = lag_time
            df['max_yield'] = max_yield
            df['max_growth_rate'] = growth_rate
            df = df.drop('zwieter', axis=1)
            fit_df.append(df)
        
        self.data = pd.concat(fit_df)

    def plot_fitted_df(self, xData, yData, popt, name):
        x_fit = np.arange(0, 100, 0.1)

        #Plot the fitted function
        fig, ax = plt.subplots()
        plt.plot(x_fit, self.zwietering_model(x_fit, *popt), 'bo', markersize=1, label='fitted model')
        plt.plot(xData, yData, 'ro', markersize=1, label = 'Data')
        plt.xlim(0, 100)
        plt.ylabel('ln(OD/OD0)')
        plt.xlabel('Time')
        plt.title(name)
        ax.legend()

        plot_path = os.path.join(self.folder, "Experiment_plots", "Zwietering")
        Path(plot_path).mkdir(parents=True, exist_ok=True)

        plt.savefig(f"{plot_path}/Zwietering_{name}.png")
        plt.close()

       
    def align_data(self):
        self.aligned_data = af.align_df(self.data, self.filter_value)
    
    def calculate_lag_phase(self, df):

        annotated_df = []
        df = df.copy()

        for name, df in df.groupby('variable'):
            current_values = self.new_df[self.new_df['name'] == name]
            df['growth_phase'] = 'string'
            df.loc[df['Time'] < current_values['lt'].values[0], 'growth_phase'] = 'lag_phase'
            annotated_df.append(df)

        annotated_df = pd.concat(annotated_df)

        return annotated_df

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
    
    def calculate_exponential_phase(self, df):

        annotated_df = []
        df = df.copy()

        for name, df in df.groupby('variable'):

            df_no_lag_phase = df[df['growth_phase'] != 'lag_phase']
            end_lag_phase = df_no_lag_phase.index[0]-1
            sorted_gr_index = np.argsort(df_no_lag_phase['growth_rate'].values)
            sorted_gr = df_no_lag_phase['growth_rate'].values[sorted_gr_index]

            sorted_gr = sorted_gr[~np.isnan(sorted_gr)]

            max_growth_rate = np.mean(sorted_gr[-8:])
            index_end_exponential_growth = df[df['growth_rate'] > max_growth_rate].index[-1]

            df.loc[end_lag_phase:index_end_exponential_growth, 'growth_phase'] = 'exponential_phase'
            annotated_df.append(df)

        annotated_df = pd.concat(annotated_df)
        return annotated_df
    
    def calculate_post_exponential_phase(self, df):
        annotated_df = []
        df = df.copy()
        for name, df in df.groupby('variable'):
            df_end_exponential_index = df[df['growth_phase'] == 'exponential_phase']
            
            if not df_end_exponential_index.empty:
                df_end_exponential_index = df_end_exponential_index.index[-1] + 1 
            

            df_start_stationary = df[(df['OD']> 0.05) & (df['growth_rate'] < 0.01)]
            if not df_start_stationary.empty:
                df_start_stationary = df_start_stationary.index[0]
                df.loc[df_start_stationary:, 'growth_phase'] = "stationary"
            else:
                df_start_stationary = df.index[-1]
            
            df.loc[df_end_exponential_index:df_start_stationary, 'growth_phase'] = "post-exponential"
            
            annotated_df.append(df)
        
        annotated_df = pd.concat(annotated_df)
        return annotated_df
    

    def calculate_growth_phase(self):
        self.data = self.calculate_lag_phase(self.data)
        self.data = self.calculate_growth_rate(self.data)
        self.data = self.calculate_exponential_phase(self.data)
        self.data = self.calculate_post_exponential_phase(self.data)
        # self.data = self.calculate_stationary_phase(self.data)

    def visualize_growth_rate(self):
        fig, ax = plt.subplots()
        sns.scatterplot(data = df, x='Time', y='log(OD)', hue='growth_phase', ax=ax)
        plot_path = os.path.join(self.folder, "Experiment_plots", "growth_phase")
        Path(plot_path).mkdir(parents=True, exist_ok=True)
        ax.set_title(self.name)
        plt.savefig(f"{plot_path}/growth_phase.png")
        plt.close()

# now that I have the data I need I will align the dataframes then split them up
    def split_data_frames(self, df):
        df_mz = df[df['Group'].str[0:2] == 'MZ'].reset_index(drop=True).add_prefix("mz1_")
        cols_to_drop = ['Time', 'osmolarity'] # I am dropping these because they are identical between dataframes.
        df_wt = df[df['Group'].str[0:2] == 'WT'].drop(columns= cols_to_drop).reset_index(drop=True).add_prefix("wt_")
        return df_wt, df_mz

    def subtract_wt(self):
        # Now I should split self.data into containing MZ and WT
        wt_df, mz_df = self.split_data_frames(self.aligned_data)

        subtracted_df = []
        for wt_name, wt_df_loop in wt_df.groupby('wt_variable'):
            print(wt_name)
            for mz_name, mz_df_loop in mz_df.groupby('mz1_variable'):
                if mz_name[-6:] == wt_name[-6:]:
                    print(f"{mz_name} matches {wt_name}")
                    mz_df_loop = mz_df_loop.reset_index(drop=True).copy()
                    wt_df_loop = wt_df_loop.reset_index(drop=True).copy()
                    subtract_col = mz_df_loop['mz1_GFP/OD'] - wt_df_loop['wt_GFP/OD']
                    mz_df_loop['normalised_GFP/OD'] = subtract_col
                    mz_df_loop['wt_OD'] = wt_df_loop['wt_OD']
                    mz_df_loop['wt_variable'] = wt_df_loop['wt_variable']
                    mz_df_loop['wt_GFP'] = wt_df_loop['wt_GFP']
                    mz_df_loop['wt_log(OD)'] = wt_df_loop['wt_log(OD)']
                    mz_df_loop['wt_growth_rate'] = wt_df_loop['wt_growth_rate']
                    mz_df_loop['wt_Group'] = wt_df_loop['wt_Group']
                    mz_df_loop['wt_lag_time'] = wt_df_loop['wt_lag_time']
                    mz_df_loop['wt_max_growth_rate'] = wt_df_loop['wt_max_growth_rate']
                    mz_df_loop['wt_max_yield'] = wt_df_loop['wt_max_yield']
                    subtracted_df.append(mz_df_loop)
        self.normalized_df = pd.concat(
            subtracted_df).dropna().reset_index(drop=True)
        
    


experiment8 = Experiment(media='M63_Glu',
                                solute='Sucrose',
                                temperature='37',
                                date='2020-10-23',
                                folder='Data/20201023_m63Glu_37C_Sucrose',
                                plot=False)



df = experiment8.list_of_repeats[0].data

sns.scatterplot(data = df, x='Time', y='log(OD)', hue='growth_phase')


df_gfp = df[df['Group'].str[0:2] == 'MZ'].reset_index(drop=True).add_prefix("mz1_")
cols_to_drop = ['Time', 'GFP/OD', 'Group', 'osmolarity', 'variable'] # I am dropping these because they are identical between dataframes.
df_wt = df[df['Group'].str[0:2] == 'WT'].drop(columns= cols_to_drop).reset_index(drop=True).add_prefix("wt_")


merged_df = pd.concat([df_gfp, df_wt], axis=1)

df = experiment8.list_of_repeats[0]

df.normalized_df.groupby('mz1_variable').mean()['normalised_GFP/OD'].plot.('mz1_variable')
