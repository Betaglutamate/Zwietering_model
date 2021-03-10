import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

all_experiments = pickle.load(open("experiments_2021-03-10.p", "rb" ) )

all_experiments_dataframe = []


for experiment in all_experiments:
    experiment.experiment_df['experiment'] = '_'.join([experiment.name, experiment.solute])
    all_experiments_dataframe.append(experiment.experiment_df)


for experiment in all_experiments:
    sns.scatterplot(x = "osmolarity", y = "GrowthRate", hue='experiment', data = nacl_final_df)        


final_df = pd.concat(all_experiments_dataframe).reset_index(drop=True)

#Sucrose containing df
sucrose_final_df = final_df[final_df['experiment'].str.contains('Sucrose', regex=False)]

#Sucrose containing df
nacl_final_df = final_df[final_df['experiment'].str.contains('NaCl', regex=False)]

#plot max_GFP vs MAXOD
'''
Things I need to do.
1. for everyexperiment for every repeat for every variable get the calculated growthj rate and add it into the df

2. for every experiment for every variable get the maxyield and the max growthrate
'''



fig, ax  = plt.subplots(figsize = (11,7))
           
sns.scatterplot(ax = ax, x = "osmolarity", y = "GrowthRate", hue='experiment', data = nacl_final_df)        
ax.set(title = "Osmolarity vs Growth Rate NaCl", ylabel = 'GrowthRate', xlabel = 'Osmolarity')


