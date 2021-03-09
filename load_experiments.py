import model
import warnings
from concurrent.futures import ThreadPoolExecutor
import pickle
import datetime
warnings.filterwarnings("ignore")


def run_experiment1():
    experiment1 = model.Experiment(media='M63_Gly',
                               osmolyte='Sucrose',
                               temperature='37',
                               date='2020-09-10',
                               folder='Data/20200910_m63gly_37c_Sucrose',
                               plot=True)
    return experiment1


def run_experiment2():
    experiment2 = model.Experiment(media='M63_Glu_CAA',
                               osmolyte='Sucrose',
                               temperature='30',
                               date='2020-09-29',
                               folder='Data/20200929_m63GluCAA_30c_Sucrose',
                               plot=True)
    return experiment2

list_test = []
with ThreadPoolExecutor(max_workers=3) as executor:
    experiment1 = executor.submit(run_experiment1)
    experiment2 = executor.submit(run_experiment2)
    list_test.append(run_experiment1)
    list_test.append(run_experiment2)

for future in list_test:
    result = future.result()
    print(result)