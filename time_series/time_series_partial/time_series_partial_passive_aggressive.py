from sktime.transformations.panel.rocket import Rocket
import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import time_series_function_partial as tsf


main_df = pd.read_csv('all_data.csv')
length_window = 10
plot_fit = False


#load rocket and classifier
try:
    with open('20210430_PA_trained.pickle', 'rb') as handle:
        model = pickle.load(handle)
        rocket, classifier = model
    print('model found and loaded')
except FileNotFoundError:
    print('no existing model found initializing fresh instance of rocket and classifier')
    rocket = Rocket()
    classifier = PassiveAggressiveClassifier()
# main_df_try = main_df_list[0]

rocket, classifer = tsf.train_model(main_df, rocket, 30, classifier)

trained_network = [rocket, classifer]


print('classification finished saving instance of rocket and classifier')

with open('20210430_PA_trained.pickle', 'wb') as handle:
    pickle.dump(trained_network, handle, protocol=4)

print('trained model saved')

#initialize rocket once only

for name, df in main_df.groupby('experiment'):
    tsf.create_fitted_plots(main_df, name, classifier, rocket, length_window)


