from analysisFunctions import normalize_plate, align_df, generate_plots, calculate_regression
import pandas as pd
import numpy as np
from scipy import stats

files_to_analyze = []

import os
for root, dirs, files in os.walk("Data"):
    for filename in files:
        if filename.endswith(".xlsx"):
            files_to_analyze.append({"root": root, "filename": filename})

def analyze_plate(file, **kwargs):
    keywords = kwargs

    root = file['root']
    filename = file['filename']

    plate_path = os.path.join(root, filename)
    plate_normalized = normalize_plate(plate_path)

    # Align the dataframe to a specific value for individual values

    a = plate_normalized['OD'].melt(id_vars="Time (min)")
    a = a.rename(columns = dict(zip(a.columns, ["Time", "variable", "OD"])))

    b = plate_normalized['GFP'].melt(id_vars="Time (min)")
    b = b.rename(columns = dict(zip(b.columns, ["Time", "variable", "GFP"])))

    merged = a.merge(b)

    #OK its in the long format now to align it to OD

    split = merged.groupby('variable')    
    split_df = [split.get_group(x) for x in split.groups]

    aligned_df_long = []

    for df in split_df:
        aligned_df = align_df(df)
        aligned_df_long.append(aligned_df)

    aligned_df_long = pd.concat(aligned_df_long)    
    aligned_df_long['Group'] = aligned_df_long['variable'].apply(lambda x: x[0:7])
    aligned_df_long['GFP/OD'] = aligned_df_long['GFP'] / aligned_df_long['OD']
    aligned_df_long['log(OD)'] = np.log(aligned_df_long['OD'])

    if keywords.get('plot'):
        print("generating plots")
        generate_plots(aligned_df_long, root)

    aligned_df_long = calculate_regression(aligned_df_long)

    return aligned_df_long


test = analyze_plate(files_to_analyze[0], plot = False)

split = test.groupby('variable')    
split_df = [split.get_group(x) for x in split.groups]

(
    gg.ggplot(split_df[8][20:]) +
    gg.aes(x="Time", y= "GrowthRate") +
    gg.geom_point()

)

a = split_df[8]['GrowthRate'].values

diff_length = []

for num in range(len(a)-8):
    val1 = a[num]
    val2 = a[num+2]
    val3 = a[num+3]
    val4 = a[num+4]
    val5 = a[num+5]
    val6 = a[num+6]
    val7 = a[num+7]
    val8 = a[num+8]

    dif1 = (val1-val2)**2
    dif2 = (val1-val3)**2
    dif3 = (val1-val4)**2
    dif4 = (val1-val5)**2
    dif5 = (val1-val6)**2
    dif6 = (val1-val7)**2
    dif7 = (val1-val8)**2
    final_dif = dif1 + dif2 + dif3 + dif4 + dif5 + dif6 + dif7
    diff_length.append(final_dif)

diff_length = pd.DataFrame({"diff": diff_length})

final = pd.concat([split_df[8],diff_length], axis=1)
final.loc[final['diff'] < 0.0005]


(
    gg.ggplot(final) +
    gg.aes(x="Time", y= "diff") +
    gg.geom_point()+
    gg.ylim(0, 0.0025)

)

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





