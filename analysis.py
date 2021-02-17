from analysisFunctions import normalizePlate, alignDf, generatePlots, calculateRegression
import pandas as pd
import numpy as np
from scipy import stats

filesToAnalyze = []

test = analyzePlate(filesToAnalyze[1])

import os
for root, dirs, files in os.walk("Data"):
    for filename in files:
        if filename.endswith(".xlsx"):
            filesToAnalyze.append({"root": root, "filename": filename})

def analyzePlate(file):

    root = file['root']
    filename = file['filename']

    plate1Path = os.path.join(root, filename)
    plate1Normalized = normalizePlate(plate1Path)

    # Align the dataframe to a specific value for individual values

    a = plate1Normalized['OD'].melt(id_vars="Time (min)")
    a = a.rename(columns = dict(zip(a.columns, ["Time", "variable", "OD"])))

    b = plate1Normalized['GFP'].melt(id_vars="Time (min)")
    b = b.rename(columns = dict(zip(b.columns, ["Time", "variable", "GFP"])))

    merged = a.merge(b)

    #OK its in the long format now to align it to OD

    split = merged.groupby('variable')    
    splitDf = [split.get_group(x) for x in split.groups]

    alignedDfLong = []

    for df in splitDf:
        alignedDf = alignDf(df)
        alignedDfLong.append(alignedDf)

    alignedDfLong = pd.concat(alignedDfLong)    
    alignedDfLong['Group'] = alignedDfLong['variable'].apply(lambda x: x[0:7])
    alignedDfLong['GFP/OD'] = alignedDfLong['GFP'] / alignedDfLong['OD']
    alignedDfLong['log(OD)'] = np.log(alignedDfLong['OD'])

    generatePlots(alignedDfLong, root)
    alignedDfLong = calculateRegression(alignedDfLong)

    return alignedDfLong

import time

def calculateRegression(df):
    """
    This function will take the longDf and calculate a rolling window
    fit for growth rate
    """
    windowSize = 8

    regress_list = []
    for length in range(len(df)):
        res = stats.linregress(
            x = df['X'].values[length:length+windowSize],
            y = df['Y'].values[length:length+windowSize]
        )
        regress_list.append((res.slope*60))
    
    df['regression'] = regress_list

    return df

start = time.time()

x = np.random.random(10000)
y = np.random.random(10000)
test = pd.DataFrame({"X": x, "Y": y})
calculateRegression(test)

end = time.time()

print(end-start)





