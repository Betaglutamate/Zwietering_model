from numpy.testing._private.utils import temppath
import analysisFunctions as af
from pathlib import Path
import os
import plotnine as gg


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
            list_of_repeats.append(temp_plate)
            self.list_of_repeats = list_of_repeats


class Plate(Experiment):

    def __init__(self, media, osmolyte, temperature, date, folder, repeat_number, data):
        super().__init__(media, osmolyte, temperature, date, folder)
        self.repeat_number = repeat_number
        self.data = data

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
            OD_group_plot = (
                gg.ggplot(df) +
                gg.aes(x='Time', y='OD', color='variable') +
                gg.geom_point() +
                gg.ggtitle(df['Group'].values[0])
            )
            save_string = f"OD_{df['Group'].values[0]}_{self.repeat_number}.png"
            gg.ggsave(OD_group_plot, os.path.join(plot_path, save_string))

        for df in split_df:
            LN_group_plot = (
                gg.ggplot(df) +
                gg.aes(x='Time', y='log(OD)', color='variable') +
                gg.geom_point() +
                gg.ggtitle(df['Group'].values[0])
            )
            save_string = f"lnOD_{df['Group'].values[0]}_{self.repeat_number}.png"
            gg.ggsave(LN_group_plot, os.path.join(plot_path, save_string))
        

    def calculate_max_growth_rate(self):
        split = self.data.groupby('variable')
        split_df = [split.get_group(x) for x in split.groups]
        max_growth_rate_dict = {}

        for temp_df in split_df:
            new_df = temp_df[temp_df['OD'] > 0.02]
            if new_df.empty:
                max_growth_rate = 0
            else:
                max_growth_rate = max(new_df['GrowthRate'])

            max_growth_rate_dict[temp_df['variable'].values[0]] = max_growth_rate

        self.max_growth_rate = max_growth_rate_dict

    def plot_growth_rate(self):
        plot_path = os.path.join(self.folder, "plots", self.repeat_number, "GrowthRate")
        Path(plot_path).mkdir(parents=True, exist_ok=True)

        split = self.data.groupby('variable')
        split_df = [split.get_group(x) for x in split.groups]

        for df in split_df:
            current_variable = df['variable'].values[0]
            LN_group_plot = (
                gg.ggplot(df) +
                gg.aes(x='Time', y='GrowthRate', color='variable') +
                gg.geom_point() +
                gg.geom_hline(yintercept = self.max_growth_rate[current_variable]) +
                gg.ggtitle(current_variable)
            )
            save_string = f"GR_{current_variable}_{self.repeat_number}.png"
            gg.ggsave(LN_group_plot, os.path.join(plot_path, save_string))
    
    def subtract_wt(self):
        #Now I should split self.data into containing MZ and WT
        wt_df = self.data[self.data['variable'].str.upper().str.match('WT')]
        mz_df = self.data[self.data['variable'].str.upper().str.match('MZ')]

        #Then I need to match the last 6 chars of the variable and subtract
        split = wt_df.groupby('variable')
        split_wt = [split.get_group(x) for x in split.groups]

        split = mz_df.groupby('variable')
        split_mz = [split.get_group(x) for x in split.groups]

        for wt_variable in split_wt:
            for mz_variable in split_mz:
                if mz_variable['variable'].values[0][-6:] == wt_variable['variable'].values[0][-6:]:
                    mz_variable = mz_variable.reset_index(drop=True)
                    wt_variable = wt_variable.reset_index(drop=True)
                    subtract_col = mz_variable['GFP/OD'].reset_index(drop=True) - wt_variable['GFP/OD'].reset_index(drop=True)
                    mz_variable['normalised_GFP/OD'] = subtract_col
                    mz_variable['wt_variable'] = wt_variable['variable']




    
