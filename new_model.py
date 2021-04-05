import run_plate as af
from pathlib import Path
import os
import matplotlib.pyplot as plt
import plotnine as gg
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import date
from scipy.optimize import curve_fit



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
        self.filter_value = 0.00025
        self.length_exponential_phase = 8
        print(f"processing {self.name}")
        self.clean_data()

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
                               filter_value=self.filter_value,
                               length_exponential_phase=self.length_exponential_phase)
            temp_plate.fit_df()
            list_of_repeats.append(temp_plate)
            self.list_of_repeats = list_of_repeats

  



class Plate():

    def __init__(self, media, solute, temperature, date, folder, repeat_number, data, filter_value, length_exponential_phase):

        self.name = media
        self.solute = solute
        self.temperature = temperature
        self.date = date
        self.folder = folder
        self.repeat_number = repeat_number
        self.data = data
        self.filter_value = filter_value
        self.length_exponential_phase = length_exponential_phase

    def zwietering_model(self, t, A, Kz, Tlag):
        return  A*np.exp(-np.exp(((np.e*Kz)/A)*(Tlag-t)+1))
        
    def fit_df(self):
        
        names = []
        gr = []
        my = []
        lt = []
        
        for name, df in self.data.groupby('variable'):


            df['zwieter'] = (np.log(df['OD'] / (df['OD'].values[0])))
            xData = df['Time']
            yData = df['zwieter']
            p0_test = [1, 0.8, 3]

            popt, _ = curve_fit(f=self.zwietering_model, xdata=xData, ydata=yData, p0=p0_test, maxfev = 1000)


            x_fit = np.arange(0, 100, 0.1)

        #Plot the fitted function
            plt.plot(x_fit, self.zwietering_model(x_fit, *popt), 'bo', markersize=1)
            plt.plot(xData, yData, 'ro', markersize=1)
            plt.xlim(0, 100)
            plt.ylabel('ln(OD/OD0)')

            plt.xlabel('Time')
            plt.title(name)

            plot_path = os.path.join(self.folder, "Experiment_plots")
            Path(plot_path).mkdir(parents=True, exist_ok=True)

            plt.savefig(f"{plot_path}/Zwietering_{name}.png")
            plt.close()

            growth_rate = popt[1]
            max_yield = popt[0]
            lag_time = popt[2]
            max_yield = np.exp(max_yield)*df['OD'].values[0]

            names.append(name)
            lt.append(lag_time)
            my.append(lag_time)
            gr.append(growth_rate)
            new_df = pd.DataFrame({"name": names, "gr":gr, "my":my, "lt": lt})
            self.new_df = new_df



experiment8 = Experiment(media='M63_Glu',
                                solute='Sucrose',
                                temperature='37',
                                date='2020-10-23',
                                folder='Data/20201023_m63Glu_37C_Sucrose',
                                plot=False)
