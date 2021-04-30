from numba.cuda.simulator import kernel
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Perceptron
import pickle

import time_series_function_partial as tsf


main_df = pd.read_csv('all_data.csv')
length_window = 30

main_df_list = [df for _, df in main_df.groupby('experiment')]

testing_df = main_df_list[0]

#load rocket and classifier
try:
    with open('20210430_perceptron_trained.pickle', 'rb') as handle:
        model = pickle.load(handle)
        rocket, classifier = model
    print('model found and loaded')
except FileNotFoundError:
    print('no existing model found initializing fresh instance of rocket and classifier')
    rocket = Rocket()
    classifier = Perceptron(penalty="elasticnet")
# main_df_try = main_df_list[0]

rocket, classifer = tsf.train_model(main_df, rocket, 30, classifier)

trained_network = [rocket, classifer]


print('classification finished saving instance of rocket and classifier')

with open('20210430_perceptron_trained.pickle', 'wb') as handle:
    pickle.dump(trained_network, handle, protocol=4)

print('trained model saved')

#initialize rocket once only
testing_df = main_df_list[0]

tsf.generate_fitted_plots(testing_df, classifier, rocket, length_window)




