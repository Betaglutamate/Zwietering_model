import pandas as pd
import numpy as np
import csv

from pandas.core.indexes.base import Index


def normalizePlate(pathToExcel):
# so first we want to open and analyse a single file

    inputDf = pd.read_excel(plate1Path)


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
    test= []
    lenToFindmeanOdStart = 10

    for i in inputDfOd.iloc[0:lenToFindmeanOdStart].values:
        for x in (i[1:]):
            test.append(x) 

    allList = []

    for i in range(len(inputDfOd.columns)-1):
        allList.append(test[i::60])
    
    subtractInitialOd = []

    for initialOd in allList:
        temp = np.mean(initialOd)
        subtractInitialOd.append(temp)
    
    
    for num, i in enumerate(subtractInitialOd):


        print(num)
        inputDfOd.iloc[(num+1)] = inputDfOd.iloc[(num+1)].apply(lambda row: row - i)

    firstRowGfp = inputDfGfp.iloc[[0]].values[0]
    finalGfp = inputDfGfp.apply(lambda row: row - firstRowGfp, axis=1)

    return {"OD": finalOd, "GFP": finalGfp}


    for num, i in enumerate(subtractInitialOd):
        test.iloc[(num+1)] = test.iloc[(num+1)].apply(lambda row: row - i)


        



    


