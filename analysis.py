import model
import warnings
from concurrent.futures import ProcessPoolExecutor
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

def run_experiment3():
    experiment3 = model.Experiment(media='M63_Glu_CAA',
                               osmolyte='Sucrose',
                               temperature='37',
                               date='2020-10-02',
                               folder='Data/20201002_m63GluCAA_37C_Sucrose',
                               plot=True)
    return experiment3


def run_experiment4():
    experiment4 = model.Experiment(media='M63_Gly_Betaine',
                                osmolyte='Sucrose',
                                temperature='37',
                                date='2020-10-06',
                                folder='Data/20201006_m63Gly_Betaine_37C_Sucrose',
                                plot=True)
    return experiment4

def run_experiment5():
    experiment5 = model.Experiment(media='M63_Glu_CAA_Betaine',
                                osmolyte='Sucrose',
                                temperature='37',
                                date='2020-10-09',
                                folder='Data/20201009_m63GluCAA_Betaine_37C_Sucrose',
                                plot=True)
    return experiment5

def run_experiment6():
    experiment6 = model.Experiment(media='M63_Glu_CAA',
                                osmolyte='NaCl',
                                temperature='37',
                                date='2020-10-13',
                                folder='Data/20201013_m63GluCAA_37C_NaCl',
                                plot=True)
    return experiment6

def run_experiment7():
    experiment7 = model.Experiment(media='M63_Glu_CAA_Betaine',
                                osmolyte='NaCl',
                                temperature='37',
                                date='2020-10-19',
                                folder='Data/20201019_m63GluCAA_Betaine_37C_NaCl',
                                plot=True)
    return experiment7

def run_experiment8():
    experiment8 = model.Experiment(media='M63_Glu',
                                osmolyte='Sucrose',
                                temperature='37',
                                date='2020-10-23',
                                folder='Data/20201023_m63Glu_37C_Sucrose',
                                plot=True)
    return experiment8

def run_experiment9():
    experiment9 = model.Experiment(media='M63_Glu',
                               osmolyte='Sucrose',
                               temperature='42',
                               date='2020-10-27',
                               folder='Data/20201027_m63Glu_42C_Sucrose',
                               plot=True)
    return experiment9

def run_experiment10():
    experiment10 = model.Experiment(media='M63_Man',
                                osmolyte='Sucrose',
                                temperature='37',
                                date='2020-10-27',
                                folder='Data/20201030_m63Man_37C_Sucrose',
                                plot=True)
    return experiment10

def run_experiment11():
    experiment11 = model.Experiment(media='RDM',
                                osmolyte='Sucrose',
                                temperature='37',
                                date='2020-12-04',
                                folder='Data/20201204_RDM_37C_Sucrose',
                                plot=True)
    return experiment11

def run_experiment12():
    experiment12 = model.Experiment(media='M63_Gly',
                                osmolyte='NaCl',
                                temperature='37',
                                date='2021-01-19',
                                folder='Data/20210119_m63Gly_37C_NaCl',
                                plot=True)
    return experiment12

def run_experiment13():
    experiment13 = model.Experiment(media='M63_Gly_Betaine',
                                osmolyte='NaCl',
                                temperature='37',
                                date='2021-01-25',
                                folder='Data/20210125_m63Gly_Betaine_37C_NaCl',
                                plot=True)
    return experiment13

def run_experiment14():
    experiment14 = model.Experiment(media='M63_Gly',
                                osmolyte='NaCl',
                                temperature='37',
                                date='2021-02-10',
                                folder='Data/20210202_m63Gly_37C_NaCl',
                                plot=True)
    return experiment14

def run_experiment15():
    experiment15 = model.Experiment(media='M63_Glu',
                                osmolyte='NaCl',
                                temperature='37',
                                date='2021-02-16',
                                folder='Data/20210216_m63Glu_37C_NaCl',
                                plot=True)
    return experiment15

list_test = []

with ProcessPoolExecutor(max_workers=3) as executor:
    experiment1 = executor.submit(run_experiment1)
    experiment2 = executor.submit(run_experiment2)
    experiment3 = executor.submit(run_experiment3)
    experiment4 = executor.submit(run_experiment4)
    experiment5 = executor.submit(run_experiment5)
    experiment6 = executor.submit(run_experiment6)

with ProcessPoolExecutor(max_workers=3) as executor:
    experiment7 = executor.submit(run_experiment7)
    experiment8 = executor.submit(run_experiment8)
    experiment9 = executor.submit(run_experiment9)
    experiment10 = executor.submit(run_experiment10)
    experiment11 = executor.submit(run_experiment11)
    experiment12 = executor.submit(run_experiment12)

with ProcessPoolExecutor(max_workers=3) as executor:

    experiment13 = executor.submit(run_experiment13)
    experiment14 = executor.submit(run_experiment14)
    experiment15 = executor.submit(run_experiment15)

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
]
    

pickle.dump(experiment_summary, open( f"experiments_{str(datetime.date.today())}.p", "wb" ) )

# pickle.load(open("save.p", "rb" ) )


# experiment3 = model.Experiment(media='M63_Glu',
#                                osmolyte='NaCl',
#                                temperature='37',
#                                date='2021-02-28',
#                                folder='Data/20210216_m63Glu_NaCl',
#                                plot=False)


#files_to_analyze = []

# for root, dirs, files in os.walk("Data"):
#     for filename in files:
#         if filename.endswith(".xlsx"):
#             files_to_analyze.append({"root": root, "filename": filename})


# for num, file in enumerate(files_to_analyze[0:7]):
#     print(file)
#     if fn.fnmatch(file['root'].lower(), '*nacl*'):
#         temp_df = af.analyze_plate(files_to_analyze[num], plot=True)
#         GR_list = af.calculate_max_growth_rate(temp_df)


#         split = temp_df.groupby('variable')
#         split_df = [split.get_group(x) for x in split.groups]

#         i=0

#         for gr, df in zip(GR_list, split_df):

#             i = i+1

#             x = sns.scatterplot(data = df, x = 'Time', y= "GrowthRate", hue="variable", edgecolor = 'none')
#             x.set(xlabel='Time (h)', ylabel='Growth Rate u-1', title=file['root'] + file['filename'])
#             x.axhline(gr, ls='--')

#             savestring = os.path.join(file['root'], str(i))

#             x.figure.savefig(savestring)
#             plt.clf()


# make multiplot

#fig, ax = plt.subplots(2,2)

# sns.scatterplot(x="Time", y = "mass", ax =ax[0][0], edgecolor = 'none', hue="Group", s=1, data=newdf)
# ax[0][0].set(xlabel='Time (h)', ylabel='weight (mg)', title='')
# ax[0][0].get_legend().remove()

# plt.tight_layout()

# handles, labels = ax[0][0].get_legend_handles_labels()
# fig.legend(handles= handles, labels =labels,
#            loc=[0.8, 0.4],   # Position of legend
#            borderaxespad=0.1)

# plt.subplots_adjust(right=0.75)


# ax.savefig("weight.png")


# five_colours = ['blue', 'red', 'green', 'yellow', 'orange']*4
# unique_variables = newdf['variable'].unique()

# colourdict = dict(zip(unique_variables, five_colours))
# newdf['colour'] = np.nan


# newdf['colour'] = newdf.apply(lambda x: colourdict.get(x.variable), axis=1)


# import seaborn as sns


# import matplotlib.pyplot as plt

# g = sns.FacetGrid(newdf, col="Group")
# g.map(sns.relplot(x="Time", y = "mass", edgecolor = 'none', hue="Group", s=1, data=newdf))
# g.add_legend()

# i = 0
# for gr, df in zip(GR_list, split_df):

#     i = i+1

#     tempPlot = (
#     gg.ggplot(df) +
#     gg.aes(x= "Time", y = "GrowthRate") +
#     gg.geom_point()+
#     gg.geom_hline(yintercept = gr)
#     )

#     savestring = f"test{i}.png"

#     gg.ggsave(tempPlot, savestring)


# test.loc[(test['Group'] == "MZ_0800") | (test['Group'] == "WT_0800")]
# idx = np.where((test['Group'] == "MZ_0800") | (test['Group'] == "WT_0800") | (test['Group'] == "MZ_0000") | (test['Group'] == "WT_0000"))

# newdf = test.loc[idx]
# newdf['mass'] = np.nan

# newdf['mass'][(newdf['Group'] == "MZ_0800") | (newdf['Group'] == "WT_0800")] =  newdf['OD'][(newdf['Group'] == "MZ_0800") | (newdf['Group'] == "WT_0800")].apply(lambda x: x*0.978)
# newdf['mass'][(newdf['Group'] == "MZ_0000") | (newdf['Group'] == "WT_0000")] =  newdf['OD'][(newdf['Group'] == "MZ_0000") | (newdf['Group'] == "WT_0000")].apply(lambda x: x*0.4)

# newdf.plot(x="Time", y="mass", kind = "scatter")

# ax  = sns.relplot(x="Time", y = "mass", edgecolor = 'none', hue="Group", s=1, data=newdf)
# ax.set(xlabel='Time (h)', ylabel='weight (mg)', title='')

# import matplotlib.pyplot as plt

# ax.savefig("weight.png")

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
