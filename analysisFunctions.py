import os
import pandas as pd
import numpy as np
import csv
import plotnine as gg
from scipy import stats




def normalizePlate(pathToExcel):
    """
    This function takes in the pathToExcel file and outputs the normalized GFP and OD values as a dict
    """
    # so first we want to open and analyse a single file

    inputDf = pd.read_excel(pathToExcel)


    inputDfOd = inputDf[45:143].transpose()
    inputDfGfp = inputDf[146:244].transpose()


    # Now I want to set the Index as Time


    inputDfOd.reset_index(inplace=True, drop=True)
    inputDfGfp.reset_index(inplace=True, drop=True)

    colsToDrop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 25,
    26, 37, 38, 49, 50, 61, 62, 73, 74, 85, 86, 87,
    88, 89, 90, 91, 92, 93, 94, 95, 96, 97]


    inputDfOd.drop(inputDfOd.columns[colsToDrop], axis=1, inplace=True)
    inputDfGfp.drop(inputDfGfp.columns[colsToDrop], axis=1, inplace=True)


    #now drop the first row as it contains our headers and drop any NA from empty data
    inputDfOd = inputDfOd.drop(inputDfOd.index[0]).dropna()
    inputDfGfp = inputDfGfp.drop(inputDfGfp.index[0]).dropna()



    # Now we need to name the columns correctly
    # Here I call the column names saved in a csv and create a list from them named colNames

    with open('Data/01_helper_data/platereaderLayout.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    colNames = data[0]

    inputDfOd.rename(columns=dict(zip(inputDfOd.columns, colNames)),  inplace=True)
    inputDfGfp.rename(columns=dict(zip(inputDfGfp.columns, colNames)),  inplace=True)


    #now find the Time in minutes
    newTime = [round(x*(7.6),2) for x in range(0, len(inputDfOd))]
    inputDfOd["Time (min)"] = newTime
    inputDfGfp["Time (min)"] = newTime

    # now I want to subtract the average of the first row from all values in the dataframe
    # OdMeanBg = inputDfOd.loc[1][2:61].mean()
    # GfpMeanBg = inputDfGfp.loc[1][2:61].mean()

    # inputDfOd.loc[:, inputDfOd.columns != 'Time (min)'] = inputDfOd.loc[:, inputDfOd.columns != 'Time (min)']- OdMeanBg
    # inputDfGfp.loc[:, inputDfGfp.columns != 'Time (min)'] = inputDfGfp.loc[:, inputDfGfp.columns != 'Time (min)']- GfpMeanBg

    firstRowOd = inputDfOd.iloc[[0]].values[0]
    finalOd = inputDfOd.apply(lambda row: row - firstRowOd, axis=1)

    firstRowGfp = inputDfGfp.iloc[[0]].values[0]
    finalGfp = inputDfGfp.apply(lambda row: row - firstRowGfp, axis=1)

    return {"OD": finalOd, "GFP": finalGfp}




def alignDf(splitDf, **kwargs):
    """
    This function takes a single experiment and aligns it to > stDev*5
    or you can pass in a kw od = num to set alignment
    """

    newTime = splitDf["Time"].values
    allignedDF = []
    sampleList = []


    stDev =  np.std(splitDf['OD'].iloc[0:10])
    OdFilterValue = stDev*5
    if kwargs:
        OdFilterValue = kwargs.get('od')
    filteredNew = splitDf.loc[splitDf['OD'] > OdFilterValue].reset_index(drop = True)
    filteredNew["Time"] = newTime[0:len(filteredNew)]

    # filteredNew = filteredNew.drop("index", axis = 1)
    return filteredNew

def generatePlots(longDf, directory):
    """
    This function takes in a long Df it should then output a graph for every
    group
    """

    split = longDf.groupby('Group')    
    splitDf = [split.get_group(x) for x in split.groups]

    for num, df in enumerate(splitDf):
        groupPlot = (
            gg.ggplot(df)+
            gg.aes(x='Time', y='OD', color='variable')+
            gg.geom_point()+
            gg.ggtitle(df['Group'].values[0])
        )
        saveString = f"test{num}.png"
        gg.ggsave(groupPlot, os.path.join(directory, saveString))


def calculateRegression(longDf):
    """
    This function will take the longDf and calculate a rolling window
    fit for growth rate
    """
    windowSize = 8
    split = longDf.groupby('Group')    
    splitDf = [split.get_group(x) for x in split.groups]
    longDf = []

    for df in splitDf:
        regress_list = []
        for length in range(len(df)):
            res = stats.linregress(
                x = df['Time'].values[length:length+windowSize],
                y = df['log(OD)'].values[length:length+windowSize]
            )
            regress_list.append((res.slope*60))
        
        df['GrowthRate'] = regress_list
        longDf.append(df)
    
    longDf = pd.concat(longDf)
    return longDf






# model = RollingOLS(endog =df['OD'].values , exog=df[['Time']],window=5)
# rres = model.fit()
# rres.params
