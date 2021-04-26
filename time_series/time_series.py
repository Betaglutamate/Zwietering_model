import pandas as pd
import time_series_function as tsf

main_df = pd.read_csv('all_data.csv')
length_window = 20

main_df['OD'] = main_df['OD']-0.005

all_experiment_list = []

for name, experiment in main_df.groupby('experiment'):
    all_experiment_list.append(experiment)

train_data = all_experiment_list[1]
test_data = all_experiment_list[2]

classifier, rocket = tsf.generate_classifier(train_data, test_data, length_window)

tsf.generate_fitted_plots(test_data, classifier, rocket, length_window)


import pickle

trained_model = [rocket, classifier]
with open('rocket.pkl', 'wb') as output:
    pickle.dump(trained_model, output, pickle.HIGHEST_PROTOCOL)