import analysisFunctions as af
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
                    files_to_analyze.append({"root": root, "filename": filename})
        
        list_of_repeats = []

        for repeat in files_to_analyze:
            filepath = os.path.join(repeat['root'], repeat['filename'])
            analyzed_plate = af.analyze_plate(filepath)
            list_of_repeats.append(analyzed_plate)           
            self.clean_repeats = list_of_repeats
    
    def generate_plots(self):
        """
        This function takes in a long Df it should then output a graph for every
        group
        """

        from pathlib import Path
        Path(os.path.join(self.folder, "plots")).mkdir(parents=True, exist_ok=True)
        for i, repeat in  enumerate(self.clean_repeats):
            split = repeat.groupby('Group')
            split_df = [split.get_group(x) for x in split.groups]

            for df in split_df:
                group_plot = (
                    gg.ggplot(df) +
                    gg.aes(x='Time', y='OD', color='variable') +
                    gg.geom_point() +
                    gg.ggtitle(df['Group'].values[0])
                )
                save_string = f"{df['Group'].values[0]}_repeat{i}.png"
                gg.ggsave(group_plot, os.path.join(self.folder, "plots", save_string))


    