import pandas as pd

main_df = pd.read_csv('experiment1_df.csv').drop(columns=['Unnamed: 0'])

#you want to create a list of wavelets for each variable

length_window = 20

def create_subcurve(single_variable):
    X_values = []
    y_values = []
    for start_window in (range(len(single_variable.index)-length_window)):
        window = single_variable.iloc[start_window: start_window+length_window]
        X = window['log(OD)']
        y = window['growth_phase'].values[0]
        X_values.append(X)
        y_values.append(y)

    return X_values, y_values


X_list = []

y_list = []

for name, variable in main_df.groupby('variable'):
    X, y = create_subcurve(variable)
    X_list.append(X)
    y_list.append(y)


# Ok no you want to split wt and GFP

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket


xTest = X_list[0]
yTest = y_list[0]

df_test_x = pd.DataFrame({"dim_0": xTest})
df_test_y = pd.DataFrame({"dim_0": yTest})

rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
rocket.fit(df_test_x)
X_train_transform = rocket.transform(df_test_x)

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
classifier.fit(X_train_transform, yTest)

##here you are seeing if it works against new data

xTest = X_list[1]
yTest = y_list[1]

df_test_x = pd.DataFrame({"dim_0": xTest})
df_test_y = pd.DataFrame({"dim_0": yTest})

X_test_transform = rocket.transform(df_test_x)

classifier.score(X_test_transform, df_test_y)


