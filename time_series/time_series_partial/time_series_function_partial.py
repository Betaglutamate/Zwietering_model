from numba.cuda.simulator import kernel
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path


from pathlib import Path
import os



def create_subcurve(single_variable, length_window):
    X_values = []
    X_log_values = []
    y_values = []
    time_values = []
    y_growth_values = []
    for start_window in (range(len(single_variable.index)-length_window)):
        window = single_variable.iloc[start_window: start_window+length_window]
        X = window['OD']
        X_log = window['log(OD)']
        y = window['growth_phase'].values[0]
        y_growth = window['growth_rate']
        time = window['Time'].values[0]
        X_values.append(X)
        y_values.append(y)
        time_values.append(time)
        y_growth_values.append(y_growth)
        X_log_values.append(X_log)

    return X_values, y_values, y_growth_values, time_values, X_log_values

def create_data(variable, window_length):
    x_od = []
    y_current_growth_phase = []
    x_growth_list = []
    time_list = []
    x_log_od = []


    X, y, y_growth, time, X_log_values = create_subcurve(variable, window_length)
    x_od.append(X)
    y_current_growth_phase.append(y)
    time_list.append(time)
    x_growth_list.append(y_growth)
    x_log_od.append(X_log_values)

    # now try for the entire data
    flat_x = [item for sublist in x_od for item in sublist]
    flat_y = [item for sublist in y_current_growth_phase for item in sublist]
    flat_growth = [item for sublist in x_growth_list for item in sublist]
    flat_log_x = [item for sublist in x_log_od for item in sublist]

    x_data = np.array(pd.Series(flat_x))
    x_log_data = np.array(pd.Series(flat_log_x))
    x_growth_data = np.array(pd.Series(flat_growth))
    growth_phase = np.array(flat_y)

    df_train_x = pd.DataFrame(
        {"dim_0": x_data, "dim_1": x_growth_data, "dim_2": x_log_data})
    
    return df_train_x, growth_phase, time_list

def train_model(df, rocket, window_length, classifier):

    variable_list = [[name, df] for name, df in df.groupby(['experiment', 'variable'])]
    name_test_df, test_df = variable_list[0]

    rocket, classifier = initialize_train_model(variable_list, window_length, rocket, classifier)

    for name, variable in variable_list:
        print(f'processing {name}')
        df_train_x, growth_phase, time_list = create_data(variable, window_length)    
        X_train_transform = rocket.transform(df_train_x)
        try:
            classifier.partial_fit(X_train_transform, growth_phase, classes=np.unique(growth_phase))
            test_model(classifier, rocket, test_df, window_length)
        except ValueError:
            print('Value error df not used for training')

    return rocket, classifier


def test_model(classifier, rocket, test_df, window_length):

    df_test_x, growth_phase, time_list = create_data(test_df, window_length)
    X_test_transform = rocket.transform(df_test_x)
    score = classifier.score(X_test_transform, growth_phase)
    print(f'Current score is {score}')


def initialize_train_model(variable_df_list, window_length, rocket, classifier):
    #see if rocket is fitted
    try:
        rocket.check_is_fitted()
    except:
        print('rocket is now being fitted')
        name, initial_df = variable_df_list[0]
        df_train_x, growth_phase, time_list = create_data(initial_df, window_length)
        rocket.fit(df_train_x)

    try:
        classifier
    except NameError:
        print('classifier not initialized')
    else:
        print('existing classifier found')
    return rocket, classifier


# ## different df test


def create_fitted_plots(experiment_df, experiment_name, classifier, rocket, window_length):

    plot_path = os.path.join('plots', experiment_name)
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    
    for name, variable in experiment_df.groupby(['experiment', 'variable']):
        try:
            x_od, y_growth_phase, x_growth, time, _ = create_subcurve(variable, window_length)

            df_test_x, growth_phase, time_list = create_data(variable, window_length)
            X_test_transform = rocket.transform(df_test_x)

            transformed = classifier.predict(X_test_transform)
            xOD = np.fromiter((x.values[0] for x in x_od), float)
            xGR = np.fromiter((x.values[0] for x in x_growth), float)

            print(f'made prediction {name}')
            fig, [ax1, ax2] = plt.subplots(2)
            sns.scatterplot(x=time, y=xOD, hue=transformed, ax=ax1, s=5)
            sns.scatterplot(x=time, y=xGR, hue=transformed, ax=ax2, s=5)
            ax2.set(title='Growth Rate')
            ax1.set(title='ln(OD)')
            ax2.get_legend().remove()
            plt.suptitle(name[1])

            plt.tight_layout()
            plt.savefig(f'{os.path.join(plot_path, name[1])}.png', transparent = False, dpi=300)
            plt.close()
        except:
            print(f"couldnt plot {name}")
