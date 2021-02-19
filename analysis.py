from analysisFunctions import *
from  fitderiv import fitderiv as fd
import sys



files_to_analyze = []

import os
for root, dirs, files in os.walk("Data"):
    for filename in files:
        if filename.endswith(".xlsx"):
            files_to_analyze.append({"root": root, "filename": filename})


def calculate_growth_rate(df, OD_filter = 0.02):
    sys.stdout = open(os.devnull, 'w') #block fitderiv from printing
    split = test.groupby('variable')    
    split_df = [split.get_group(x) for x in split.groups]

    growth_rate_list = []
    
    for df in split_df:
        OD = df['OD'].values[df['OD'].values>OD_filter]
        Time = df['Time'].values[df['OD'].values>OD_filter]

        q = fd.fitderiv(Time, OD)
        temp_gr = {"Sample": df['variable'].values[0], "Growth_Rate": q.ds.get('max df')}
        growth_rate_list.append(temp_gr)
    
    sys.stdout = sys.__stdout__ #re enable print
    
    return growth_rate_list

test = analyze_plate(files_to_analyze[3], plot = False)

fitderivGR = calculate_growth_rate(test)

split = test.groupby('variable')    
split_df = [split.get_group(x) for x in split.groups]

testing = split_df[59].reset_index(drop=True)

OD = testing['OD'].values[testing['OD'].values>0.02]
Time = testing['Time'].values[testing['OD'].values>0.02]



# def find_flat(df):

#     seq = df.dropna()

#     for num in range(len(seq[:-8])):
#         bunch = df[num:num+8]
#         if all(numbers <= 0.012 for numbers in bunch):
#             return min(bunch.index)


# def calculate_max_growth_rate(df):
#     df['smoothed_GR'] = df['GrowthRate'].rolling(8, win_type = "gaussian").mean(std=0.1)
    
#     feel_terrain_list = []
#     for num, val in enumerate(df.smoothed_GR.values[4:-5]):
#         value1 = df.smoothed_GR.values[num -4]
#         value2 = df.smoothed_GR.values[num -3]
#         value3 = df.smoothed_GR.values[num -2]
#         value4 = df.smoothed_GR.values[num -1]
#         value5 = df.smoothed_GR.values[num]
#         value6 = df.smoothed_GR.values[num + 1]
#         value7 = df.smoothed_GR.values[num + 2]
#         value8 = df.smoothed_GR.values[num + 3]
#         value9 = df.smoothed_GR.values[num + 4]

#         sum_squared = (
#             ((value5-value1)**2)*0.25 +
#             ((value5-value2)**2)*0.5 +
#             ((value5-value3)**2)*0.75 +
#             ((value5-value4)**2) +
#             ((value5-value6)**2) +
#             ((value5-value7)**2)*0.75 +
#             ((value5-value8)**2)*0.5 +
#             ((value5-value9)**2)*0.25
#         )
#         feel_terrain_list.append(sum_squared)
    
#     summed = pd.DataFrame({"sumsq": feel_terrain_list})
#     df = df.reset_index()
#     df = pd.concat([df, summed], axis=1)

#     max_growth_loc = find_flat(df['sumsq'])
#     #import pdb; pdb.set_trace()
#     max_growth = np.mean(df['GrowthRate'][max_growth_loc:max_growth_loc+5])

#     return max_growth


# filterdf = [df[df["GrowthRate"]>0.2] for df in split_df]

# GR = [calculate_max_growth_rate(x) for x in split_df]



# new_test = test.groupby('variable').apply(calculate_regression_growth_rate).reset_index(drop=True)
i = 0
for gr, df in zip(fitderivGR, split_df):

    i = i+1

    tempPlot = (
    gg.ggplot(df) +
    gg.aes(x= "Time", y = "GrowthRate") +
    gg.geom_point()+
    gg.xlim(0,15)+
    gg.geom_hline(yintercept = gr["Growth_Rate"])
    )

    savestring = f"test{i}.png"

    gg.ggsave(tempPlot, savestring)



# #best so far bohmaN

# (
#     gg.ggplot(split_df[48]) +
#     gg.aes(x= "Time", y = "GrowthRate") +
#     gg.geom_point()+
#     gg.xlim(0, 1000)+
#     gg.geom_hline(yintercept = 0.20)
# )

# # meanofstdev = test['stdev'].rolling(16, win_type="gaussian").mean(std=1)#16 for 2 hours
# # meanofstdev.rename("meanstdev", inplace = True)



# import time

# def calculate_regression(df):
#     """
#     This function will take the longDf and calculate a rolling window
#     fit for growth rate
#     """
#     window_size = 8

#     regress_list = []
#     for length in range(len(df)):
#         res = stats.linregress(
#             x = df['X'].values[length:length+window_size],
#             y = df['Y'].values[length:length+window_size]
#         )
#         regress_list.append((res.slope*60))
    
#     df['regression'] = regress_list

#     return df

# start = time.time()

# x = np.random.random(10000)
# y = np.random.random(10000)
# test = pd.DataFrame({"X": x, "Y": y})
# calculate_regression(test)

# end = time.time()

# print(end-start)




