import pandas as pd
import time_series_function as tsf

main_df = pd.read_csv('all_data.csv')
test_df = pd.read_csv('experiment8_df.csv').drop(columns=['Unnamed: 0'])
length_window = 20

all_experiment_list = []

for name, experiment in main_df.groupby('experiment'):
    all_experiment_list.append(experiment)


classifier, rocket = tsf.generate_classifier(all_experiment_list[1], all_experiment_list[7], length_window)

tsf.generate_fitted_plots(test_df, classifier, rocket, length_window)


import pickle

trained_model = [rocket, classifier]
with open('rocket.pkl', 'wb') as output:
    pickle.dump(trained_model, output, pickle.HIGHEST_PROTOCOL)