from analysisFunctions import HelperFunctions as hf
import pandas as pd
import numpy as np
from scipy import stats
import plotnine as gg

files_to_analyze = []

import os
for root, dirs, files in os.walk("Data"):
    for filename in files:
        if filename.endswith(".xlsx"):
            files_to_analyze.append({"root": root, "filename": filename})


test = hf.analyze_plate(files_to_analyze[0], plot = False)


max_growth = test.groupby('variable').apply(hf.calculate_max_growth_rate)    

def calculate_max_growth_rate(df):
        df['smoothed_GR'] = df['GrowthRate'].rolling(8, win_type = "gaussian").mean(std=1)
        df['smoothed_std'] = (df['GrowthRate'].rolling(8).std())
        df['adjusted'] =  df['smoothed_std']/df['smoothed_GR']
        run_length, run_index = hf.longest_run(df['smoothed_GR'].values[0:100])
        max_growth_rate = np.mean(df['smoothed_GR'].values[run_index: run_index+run_length])

        return df


testing = calculate_max_growth_rate(split_df[59])


new_test = test.groupby('variable').apply(calculate_regression_growth_rate).reset_index(drop=True)

split = new_test.groupby('variable')    
split_df = [split.get_group(x) for x in split.groups]


(
    gg.ggplot(testing) +
    gg.aes(x= "Time", y = "adjusted") +
    gg.geom_point()+
    gg.xlim(0,1000)+
    gg.ylim(0,0.5)
)

#best so far bohmaN

(
    gg.ggplot(split_df[48]) +
    gg.aes(x= "Time", y = "GrowthRate") +
    gg.geom_point()+
    gg.xlim(0, 1000)+
    gg.geom_hline(yintercept = 0.20)
)

# meanofstdev = test['stdev'].rolling(16, win_type="gaussian").mean(std=1)#16 for 2 hours
# meanofstdev.rename("meanstdev", inplace = True)



import time

def calculate_regression(df):
    """
    This function will take the longDf and calculate a rolling window
    fit for growth rate
    """
    window_size = 8

    regress_list = []
    for length in range(len(df)):
        res = stats.linregress(
            x = df['X'].values[length:length+window_size],
            y = df['Y'].values[length:length+window_size]
        )
        regress_list.append((res.slope*60))
    
    df['regression'] = regress_list

    return df

start = time.time()

x = np.random.random(10000)
y = np.random.random(10000)
test = pd.DataFrame({"X": x, "Y": y})
calculate_regression(test)

end = time.time()

print(end-start)




