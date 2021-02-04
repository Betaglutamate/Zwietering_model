from analysisFunctions import normalizePlate
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

plate1Path ="Data/202009_M63GluCaa_Sucrose_37c/run1/20200929_sucrose_m63caaglu_30c.xlsx"

plate1Normalized = normalizePlate(plate1Path)

# calulate growth rates for all the columns

colList = plate1Normalized['OD'].columns.tolist()

individualTimeSeries = []
for column in colList[1:]:
    individualColumn = plate1Normalized['OD'][['Time (min)']].join(plate1Normalized['OD'][[column]])
    individualColumn['Sample'] = column
    individualColumn = individualColumn.rename(columns={column: "value"}, errors="raise")
    individualTimeSeries.append(individualColumn)


OdLongDf = individualTimeSeries[0]

for samples in individualTimeSeries[1:]:
    OdLongDf = OdLongDf.append(samples)

OdLongDf = OdLongDf.reset_index()

# Plot out individual OD curves.

# for label, df in OdLongDf.groupby("Sample"):
#     df.plot(x = "Time (min)", y = "value", kind="scatter", title=label)

#plot all the OD curves on one graph
from itertools import cycle
import matplotlib.colors as mcolors

cycol = cycle(mcolors.TABLEAU_COLORS)

fig, ax = plt.subplots(figsize=(8,6))
for label, df in OdLongDf.groupby("Sample"):
    df.plot(x = "Time (min)", y = "value", kind="scatter", ax=ax, label=label, c=next(cycol))
plt.legend()


# Now create the grouping variable by mutating the column
groupNames =[x[0:7] for x in OdLongDf['Sample']]
OdLongDf['Group'] = groupNames

#Plot all the single values
fig, ax = plt.subplots(figsize=(8,6))
for label, df in OdLongDf.groupby("Group"):
    df.plot(x = "Time (min)", y = "value", kind="scatter", ax=ax, label=label, c=next(cycol))
plt.legend()


#rearrange df to get the mean values out and plot mean values for it
test = OdLongDf.groupby(["Group", "Time (min)"]).mean()
new = test.drop(["index"], axis=1).unstack().transpose().reset_index().drop(["level_0"], axis=1)
groupList = set([x[0:7] for x in colList[1:]])

fig, ax = plt.subplots(figsize=(8,6))
for group in groupList:
    new.plot(x='Time (min)', y=group, ax = ax)


#Align the dataframe to a specific value
newTime = new["Time (min)"].values
allignedDF = []

for column in new.columns[1:]:
    filteredNew = new.loc[new[column] >0.001]['MZ_0000'].reset_index()
    filteredNew["Time"] = newTime[0:len(filteredNew)]
    filteredNew.reset_index()
    allignedDF.append(filteredNew)

