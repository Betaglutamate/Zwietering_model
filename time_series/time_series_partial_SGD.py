from numba.cuda.simulator import kernel
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time_series_function as tsf

from sklearn.linear_model import SGDClassifier
import pickle

#initialize rocket once only

# rocket = Rocket(num_kernels=10000)

# with open('20210429_rocket.pickle', 'wb') as handle:
#     pickle.dump(rocket, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('20210429_rocket.pickle', 'rb') as handle:
    rocket = pickle.load(handle)

# load all the data set parameters
main_df = pd.read_csv('all_data.csv')
window_length = 30

#split data into experiments and variables

variable_df_list = [df for _, df in main_df.groupby(['experiment', 'variable'])]

#initialize percepton
initial_df = variable_df_list[1]

x_od = []
y_current_growth_phase = []
x_growth_list = []
time_list = []

X, y, y_growth, time = tsf.create_subcurve(initial_df, window_length)
x_od.append(X)
y_current_growth_phase.append(y)
time_list.append(time)
x_growth_list.append(y_growth)

# now try for the entire data
flat_x = [item for sublist in x_od for item in sublist]
flat_y = [item for sublist in y_current_growth_phase for item in sublist]
flat_growth = [item for sublist in x_growth_list for item in sublist]

x_data = np.array(pd.Series(flat_x))
x_growth_data = np.array(pd.Series(flat_growth))
growth_phase = np.array(flat_y)

df_train_x = pd.DataFrame(
    {"dim_0": x_data, "dim_1": x_growth_data})

rocket.fit(df_train_x)
X_train_transform = rocket.transform(df_train_x)

classifier = SGDClassifier()
classifier.fit(X_train_transform, growth_phase)
classifier.score(X_train_transform, growth_phase)

# classifier.score(X_train_transform, growth_phase)

# classifier.predict(X_train_transform)


# train onto percepton

for df in variable_df_list[1:20]:

    x_od = []
    y_current_growth_phase = []
    x_growth_list = []
    time_list = []

    X, y, y_growth, time = tsf.create_subcurve(df, window_length)
    x_od.append(X)
    y_current_growth_phase.append(y)
    time_list.append(time)
    x_growth_list.append(y_growth)

    # now try for the entire data
    flat_x = [item for sublist in x_od for item in sublist]
    flat_y = [item for sublist in y_current_growth_phase for item in sublist]
    flat_growth = [item for sublist in x_growth_list for item in sublist]

    x_data = np.array(pd.Series(flat_x))
    x_growth_data = np.array(pd.Series(flat_growth))
    growth_phase = np.array(flat_y)

    df_train_x = pd.DataFrame(
        {"dim_0": x_data, "dim_1": x_growth_data})

    X_train_transform = rocket.transform(df_train_x)

    classifier.partial_fit(X_train_transform, growth_phase)


# test perceptron


initial_df = variable_df_list[0]

x_od = []
y_current_growth_phase = []
x_growth_list = []
time_list = []

X, y, y_growth, time = tsf.create_subcurve(initial_df, window_length)
x_od.append(X)
y_current_growth_phase.append(y)
time_list.append(time)
x_growth_list.append(y_growth)

# now try for the entire data
flat_x = [item for sublist in x_od for item in sublist]
flat_y = [item for sublist in y_current_growth_phase for item in sublist]
flat_growth = [item for sublist in x_growth_list for item in sublist]

x_data = np.array(pd.Series(flat_x))
x_growth_data = np.array(pd.Series(flat_growth))
growth_phase = np.array(flat_y)

df_train_x = pd.DataFrame(
    {"dim_0": x_data, "dim_1": x_growth_data})

X_train_transform = rocket.transform(df_train_x)

classifier.score(X_train_transform, growth_phase)

classifier.predict(X_train_transform)

#save the classifier
