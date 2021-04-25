import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket

main_df = pd.read_csv('experiment1_df.csv').drop(columns=['Unnamed: 0'])
test_df = pd.read_csv('experiment8_df.csv').drop(columns=['Unnamed: 0'])


#you want to create a list of wavelets for each variable

length_window = 20

def create_subcurve(single_variable):
    X_values = []
    y_values = []
    time_values = []
    y_growth_values = []
    for start_window in (range(len(single_variable.index)-length_window)):
        window = single_variable.iloc[start_window: start_window+length_window]
        X = window['log(OD)']
        y = window['growth_phase'].values[0]
        y_growth = window['growth_rate']
        time = window['Time'].values[0]
        X_values.append(X)
        y_values.append(y)
        time_values.append(time)
        y_growth_values.append(y_growth)

    return X_values, y_values, y_growth_values, time_values


X_list = []
y_list = []
y_growth_list = []
time_list = []

for name, variable in main_df.groupby('variable'):
    X, y, y_growth, time = create_subcurve(variable)
    X_list.append(X)
    y_list.append(y)
    time_list.append(time)
    y_growth_list.append(y_growth)


 # now try for the entire data
flat_x = [item for sublist in X_list for item in sublist]
flat_y = [item for sublist in y_list for item in sublist]
flat_growth = [item for sublist in y_growth_list for item in sublist]


# next(flat_x)
# next(flat_y)

xData = np.array(pd.Series(flat_x))
growthData = np.array(pd.Series(flat_growth))
yData = np.array(flat_y)

xData_small = xData[1::3]
yData_small = yData[1::3]
growthData_small = growthData[1::3]

df_test_x = pd.DataFrame({"dim_0": xData_small, "dim_1" : growthData_small})
df_test_y = pd.DataFrame({"dim_0": yData_small})

rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
rocket.fit(df_test_x)
X_train_transform = rocket.transform(df_test_x)

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
classifier.fit(X_train_transform, yData_small)


#Here I will test the fit of my model

x_fit_test = X_list[29]
y_fit_test = y_list[29]
y_growth_test =  y_growth_list[26]

xTime = time_list[26]

df_test_fit_x = pd.DataFrame({"dim_0": x_fit_test, "dim_1" : y_growth_test})
df_test_fit_y = pd.DataFrame({"dim_0": y_fit_test})

X_test_transform = rocket.transform(df_test_fit_x)

print(classifier.score(X_test_transform, df_test_fit_y))

#make values for plotting
xOD = np.fromiter((x.values[0] for x in y_growth_test), float)
transformed = classifier.predict(X_test_transform)

sns.scatterplot(x=xTime, y=xOD, hue=transformed)

# Ok no you want to split wt and GFP

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

## different df test

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


    
# xTest = X_list[1]
# yTest = y_list[1]
# xTime = time_list[1]

# df_test_x = pd.DataFrame({"dim_0": xTest})
# df_test_y = pd.DataFrame({"dim_0": yTest})

# X_test_transform = rocket.transform(df_test_x)

# classifier.score(X_test_transform, df_test_y)

# #make values for plotting
# xOD = np.fromiter((x.values[0] for x in xTest), float)
# transformed = classifier.predict(X_test_transform)

# sns.scatterplot(x=xTime, y=xOD, hue=transformed)