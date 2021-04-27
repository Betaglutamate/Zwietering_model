import warnings
import run_experiments_functions as re
from concurrent.futures import ProcessPoolExecutor
import pickle
import datetime
warnings.filterwarnings("ignore")
import pandas as pd

experiment1 = re.run_experiment1()


if __name__ == '__main__':


    with ProcessPoolExecutor(max_workers=5) as executor:
        experiment1 = executor.submit(re.run_experiment1)
        experiment2 = executor.submit(re.run_experiment2)
        experiment3 = executor.submit(re.run_experiment3) 
        experiment4 = executor.submit(re.run_experiment4)
        experiment5 = executor.submit(re.run_experiment5)
        experiment6 = executor.submit(re.run_experiment6)
        experiment7 = executor.submit(re.run_experiment7)
        experiment8 = executor.submit(re.run_experiment8)
        experiment9 = executor.submit(re.run_experiment9)
        experiment10 = executor.submit(re.run_experiment10)
        experiment11 = executor.submit(re.run_experiment11) 
        experiment12 = executor.submit(re.run_experiment12)
        experiment13 = executor.submit(re.run_experiment13)
        experiment14 = executor.submit(re.run_experiment14)
        experiment15 = executor.submit(re.run_experiment15)
        experiment16 = executor.submit(re.run_experiment16)
        experiment17 = executor.submit(re.run_experiment17)
        experiment18 = executor.submit(re.run_experiment18)
        experiment19 = executor.submit(re.run_experiment19)

    print("finished analysis")

    experiment_summary = [
        experiment1.result(),
        experiment2.result(),
        experiment3.result(),
        experiment4.result(),
        experiment5.result(),
        experiment6.result(),
        experiment7.result(),
        experiment8.result(),
        experiment9.result(),
        experiment10.result(),
        experiment11.result(),
        experiment12.result(),
        experiment13.result(),
        experiment14.result(),
        experiment15.result(),
        experiment16.result(),
        experiment17.result(),
        experiment18.result(),
        experiment19.result()
    ]    


    
    all_data_dataframe = []
    all_experiments_dataframe = []
    for experiment in experiment_summary:
        experiment.experiment_df['experiment'] = '_'.join([experiment.name, experiment.solute, f'{experiment.temperature}C', experiment.date])
        all_experiments_dataframe.append(experiment.experiment_df)
        experiment.full_data['experiment'] = '_'.join([experiment.name, experiment.solute, f'{experiment.temperature}C', experiment.date])
        all_data_dataframe.append(experiment.full_data)

    final_df = pd.concat(all_experiments_dataframe).reset_index(drop=True)
    full_experiment_df = pd.concat(all_data_dataframe).reset_index(drop=True)

    final_df.to_csv('final_df.csv')
    full_experiment_df.to_csv('all_data.csv')


    print("final df saved")

# experiment1 = re.run_experiment1()
# experiment2 = re.run_experiment2()
# experiment3 = re.run_experiment3() 
# experiment4 = re.run_experiment4()
# experiment5 = re.run_experiment5() 
# experiment6 = re.run_experiment6()
# experiment7 = re.run_experiment7()
# experiment8 = re.run_experiment8()
# experiment9 = re.run_experiment9()
# experiment10 = re.run_experiment10()
# experiment11 = re.run_experiment11() 
# experiment12 = re.run_experiment12()
# experiment13 = re.run_experiment13()
# experiment14 = re.run_experiment14() #error
# experiment15 = re.run_experiment15()
# experiment16 = re.run_experiment16()
# experiment17 = re.run_experiment17()
# experiment18 = re.run_experiment18()
# experiment19 = re.run_experiment19()
