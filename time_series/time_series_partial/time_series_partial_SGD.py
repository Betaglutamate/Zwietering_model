from sktime.transformations.panel.rocket import Rocket
import pandas as pd
from sklearn.linear_model import SGDClassifier
import pickle
import time_series_function_partial as tsf

main_df = pd.read_csv('all_data.csv')
length_window = 30
plot_fit = False

#load rocket and classifier
try:
    with open('20210430_sgd_trained.pickle', 'rb') as handle:
        model = pickle.load(handle)
        rocket, classifier = model
    print('model found and loaded')
except FileNotFoundError:
    print('no existing model found initializing fresh instance of rocket and classifier')
    rocket = Rocket()
    classifier = SGDClassifier(penalty="elasticnet")
# main_df_try = main_df_list[0]

rocket, classifer = tsf.train_model(main_df, rocket, 30, classifier)

trained_network = [rocket, classifer]


print('classification finished saving instance of rocket and classifier')

with open('20210430_sgd_trained.pickle', 'wb') as handle:
    pickle.dump(trained_network, handle, protocol=4)

print('trained model saved')

#initialize rocket once only

e_lost = [df for _, df in main_df.groupby('experiment')]

for name, df in main_df.groupby('experiment'):
    tsf.create_fitted_plots(main_df, classifier, rocket, length_window)


# from numba.cuda.simulator import kernel
# from sklearn.linear_model import RidgeClassifierCV
# from sklearn.pipeline import make_pipeline
# from sktime.transformations.panel.rocket import Rocket
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
# from pathlib import Path


# from sklearn.linear_model import Perceptron
# from pathlib import Path
# import os

# window_length = 30

# experiment_name = 'test'

# plot_path = os.path.join('plots', experiment_name)
# Path(plot_path).mkdir(parents=True, exist_ok=True)

# for name, variable in test_df.groupby('variable'):
#         try:
#             x_od, y_growth_phase, x_growth, time, _ = tsf.create_subcurve(variable, window_length)

#             df_test_x, growth_phase, time_list = tsf.create_data(variable, window_length)
#             X_test_transform = rocket.transform(df_test_x)

#             transformed = classifier.predict(X_test_transform)
#             xOD = np.fromiter((x.values[0] for x in x_od), float)
#             xGR = np.fromiter((x.values[0] for x in x_growth), float)

#             print(f'made prediction {name}')


#             fig, [ax1, ax2] = plt.subplots(2)
#             sns.scatterplot(x=time, y=xOD, hue=transformed, ax=ax1, s=5)
#             sns.scatterplot(x=time, y=xGR, hue=transformed, ax=ax2, s=5)
#             ax2.set(title='Growth Rate')
#             ax1.set(title='ln(OD)')
#             ax2.get_legend().remove()
#             plt.suptitle(name)

#             plt.tight_layout()
#             plt.savefig(f'{os.path.join(plot_path, name)}.png', transparent = False, dpi=300)
#             plt.close()
#         except:
#             print(f"couldnt plot {name}")