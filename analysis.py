import os
import analysisFunctions as af
from fitderiv import fitderiv as fd
import sys
import plotnine as gg

files_to_analyze = []

for root, dirs, files in os.walk("Data"):
    for filename in files:
        if filename.endswith(".xlsx"):
            files_to_analyze.append({"root": root, "filename": filename})


test = af.analyze_plate(files_to_analyze[4], plot=False)

GR_list = af.calculate_max_growth_rate(test)

split = test.groupby('variable')
split_df = [split.get_group(x) for x in split.groups]


test.loc[(test['Group'] == "MZ_0800") | (test['Group'] == "WT_0800")]
idx = np.where((test['Group'] == "MZ_0800") | (test['Group'] == "WT_0800") | (test['Group'] == "MZ_0000") | (test['Group'] == "WT_0000"))

newdf = test.loc[idx]
newdf['mass'] = np.nan

newdf['mass'][(newdf['Group'] == "MZ_0800") | (newdf['Group'] == "WT_0800")] =  newdf['OD'][(newdf['Group'] == "MZ_0800") | (newdf['Group'] == "WT_0800")].apply(lambda x: x*0.978)
newdf['mass'][(newdf['Group'] == "MZ_0000") | (newdf['Group'] == "WT_0000")] =  newdf['OD'][(newdf['Group'] == "MZ_0000") | (newdf['Group'] == "WT_0000")].apply(lambda x: x*0.4)

newdf.plot(x="Time", y="mass", kind = "scatter")

five_colours = ['blue', 'red', 'green', 'yellow', 'orange']*4
unique_variables = newdf['variable'].unique()

colourdict = dict(zip(unique_variables, five_colours))
newdf['colour'] = np.nan


newdf['colour'] = newdf.apply(lambda x: colourdict.get(x.variable), axis=1)

a =(
    gg.ggplot(newdf) +
    gg.aes(x = "Time", y = "mass", color = "Group")+
    gg.ylab("mass (g)")
    gg.geom_point()
)

gg.ggsave(a, "mass.png")


newdf.plot(x = "Time", y = "mass", kind = "scatter", c="colour")


i = 0
for gr, df in zip(GR_list, split_df):

    i = i+1

    tempPlot = (
    gg.ggplot(df) +
    gg.aes(x= "Time", y = "GrowthRate") +
    gg.geom_point()+
    gg.geom_hline(yintercept = gr)
    )

    savestring = f"test{i}.png"

    gg.ggsave(tempPlot, savestring)



# testing = split_df[59].reset_index(drop=True)

# OD = testing['OD'].values[testing['OD'].values>0.02]
# Time = testing['Time'].values[testing['OD'].values>0.02]


# def find_flat(df):

#     seq = df.dropna()

#     for num in range(len(seq[:-8])):
#         bunch = df[num:num+8]
#         if all(numbers <= 0.05 for numbers in bunch):
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
#     max_growth = np.mean(df['GrowthRate'][max_growth_loc:max_growth_loc+5])

#     return max_growth


# GR = [calculate_max_growth_rate(x) for x in split_df]

# i = 0
# for gr, df in zip(GR, split_df):

#     i = i+1

#     tempPlot = (
#     gg.ggplot(df) +
#     gg.aes(x= "Time", y = "GrowthRate") +
#     gg.geom_point()+
#     gg.xlim(0,15)+
#     gg.geom_hline(yintercept = gr)
#     )

#     savestring = f"test{i}.png"

#     gg.ggsave(tempPlot, savestring)


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
