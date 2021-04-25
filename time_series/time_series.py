import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket

main_df = pd.read_csv('experiment1_df.csv').drop(columns=['Unnamed: 0'])
test_df = pd.read_csv('experiment8_df.csv').drop(columns=['Unnamed: 0'])
length_window = 20

def create_subcurve(single_variable):
    X_values = []
    y_values = []
    time_values = []
    y_growth_values = []
    for start_window in (range(len(single_variable.index)-length_window)):
        window = single_variable.iloc[start_window: start_window+length_window]
        X = window['OD']
        y = window['growth_phase'].values[0]
        y_growth = window['growth_rate']
        time = window['Time'].values[0]
        X_values.append(X)
        y_values.append(y)
        time_values.append(time)
        y_growth_values.append(y_growth)

    return X_values, y_values, y_growth_values, time_values

def test_classifier(test_df):
    X_list = []
    y_list = []
    y_growth_list = []
    time_list = []

    for name, variable in test_df.groupby('variable'):
        X, y, y_growth, time = create_subcurve(variable)
        X_list.append(X)
        y_list.append(y)
        time_list.append(time)
        y_growth_list.append(y_growth)

    # now try for the entire data
    flat_x = [item for sublist in X_list for item in sublist]
    flat_y = [item for sublist in y_list for item in sublist]
    flat_growth = [item for sublist in y_growth_list for item in sublist]

    xData = np.array(pd.Series(flat_x))
    growthData = np.array(pd.Series(flat_growth))
    yData = np.array(flat_y)

    xData_small = xData[1::3]
    yData_small = yData[1::3]
    growthData_small = growthData[1::3]

    df_test_fit_x = pd.DataFrame({"dim_0": xData_small, "dim_1" : growthData_small})
    df_test_fit_y = pd.DataFrame({"dim_0": yData_small})

    return df_test_fit_x, df_test_fit_y

def generate_classifier(train_df, test_df):

#you want to create a list of wavelets for each variable
    X_list = []
    y_list = []
    y_growth_list = []
    time_list = []

    for name, variable in train_df.groupby('variable'):
        X, y, y_growth, time = create_subcurve(variable)
        X_list.append(X)
        y_list.append(y)
        time_list.append(time)
        y_growth_list.append(y_growth)

    # now try for the entire data
    flat_x = [item for sublist in X_list for item in sublist]
    flat_y = [item for sublist in y_list for item in sublist]
    flat_growth = [item for sublist in y_growth_list for item in sublist]

    xData = np.array(pd.Series(flat_x))
    growthData = np.array(pd.Series(flat_growth))
    yData = np.array(flat_y)

    xData_small = xData[1::3]
    yData_small = yData[1::3]
    growthData_small = growthData[1::3]

    df_train_x = pd.DataFrame({"dim_0": xData_small, "dim_1" : growthData_small})

    rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
    rocket.fit(df_train_x)
    X_train_transform = rocket.transform(df_train_x)

    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    classifier.fit(X_train_transform, yData_small)

    #Compare to another sample data here
    df_test_fit_x, df_test_fit_y = test_classifier(test_df)

    X_test_transform = rocket.transform(df_test_fit_x)

    accuracy = classifier.score(X_test_transform, df_test_fit_y)

    print(accuracy)

    return classifier, rocket

classifier, rocket = generate_classifier(main_df, test_df)



## different df test
def generate_fitted_plots(data, classifier, rocket):

    for name, variable in test_df.groupby('variable'):
        X, y, y_growth, time = create_subcurve(variable)
    
        xTest = X
        yGrowth = y_growth
        yTest = y
        xTime = time

        df_test_x = pd.DataFrame({"dim_0": xTest, "dim_1": yGrowth})
        df_test_y = pd.DataFrame({"dim_0": yTest})

        X_test_transform = rocket.transform(df_test_x)

        classifier.score(X_test_transform, df_test_y)

        #make values for plotting
        transformed = classifier.predict(X_test_transform)
        xOD = np.fromiter((x.values[0] for x in xTest), float)
        xGR = np.fromiter((x.values[0] for x in yGrowth), float)


        fig, [ax1, ax2] = plt.subplots(2)
        sns.scatterplot(x=xTime, y=xOD, hue=transformed, ax=ax1, s=5)
        sns.scatterplot(x=xTime, y=xGR, hue=transformed, ax=ax2, s=5)
        ax2.set(title='Growth Rate')
        ax1.set(title='ln(OD)')
        ax2.get_legend().remove()
        plt.suptitle(name)

        plt.tight_layout()
        plt.savefig(f'plots/od_{name}.png', transparent = False, dpi=300)
        plt.close()

generate_fitted_plots(test_df, classifier, rocket)

import pickle

trained_model = [rocket, classifier]
with open('rocket.pkl', 'wb') as output:
    pickle.dump(trained_model, output, pickle.HIGHEST_PROTOCOL)