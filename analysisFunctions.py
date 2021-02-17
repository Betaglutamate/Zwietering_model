import os
import pandas as pd
import numpy as np
import csv
import plotnine as gg
from scipy import stats




def normalize_plate(pathToExcel):
    """
    This function takes in the pathToExcel file and outputs the normalized GFP and OD values as a dict
    """
    # so first we want to open and analyse a single file

    input_df = pd.read_excel(pathToExcel)


    input_df_od = input_df[45:143].transpose()
    input_df_gfp = input_df[146:244].transpose()


    # Now I want to set the Index as Time


    input_df_od.reset_index(inplace=True, drop=True)
    input_df_gfp.reset_index(inplace=True, drop=True)

    cols_to_drop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 25,
    26, 37, 38, 49, 50, 61, 62, 73, 74, 85, 86, 87,
    88, 89, 90, 91, 92, 93, 94, 95, 96, 97]


    input_df_od.drop(input_df_od.columns[cols_to_drop], axis=1, inplace=True)
    input_df_gfp.drop(input_df_gfp.columns[cols_to_drop], axis=1, inplace=True)


    #now drop the first row as it contains our headers and drop any NA from empty data
    input_df_od = input_df_od.drop(input_df_od.index[0]).dropna()
    input_df_gfp = input_df_gfp.drop(input_df_gfp.index[0]).dropna()



    # Now we need to name the columns correctly
    # Here I call the column names saved in a csv and create a list from them named colNames

    with open('Data/01_helper_data/platereaderLayout.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    col_names = data[0]

    input_df_od.rename(columns=dict(zip(input_df_od.columns, col_names)),  inplace=True)
    input_df_gfp.rename(columns=dict(zip(input_df_gfp.columns, col_names)),  inplace=True)


    #now find the Time in minutes
    new_time = [round(x*(7.6),2) for x in range(0, len(input_df_od))]
    input_df_od["Time (min)"] = new_time
    input_df_gfp["Time (min)"] = new_time

    # now I want to subtract the average of the first row from all values in the dataframe

    firstRowOd = input_df_od.iloc[[0]].values[0]
    finalOd = input_df_od.apply(lambda row: row - firstRowOd, axis=1)

    firstRowGfp = input_df_gfp.iloc[[0]].values[0]
    finalGfp = input_df_gfp.apply(lambda row: row - firstRowGfp, axis=1)

    return {"OD": finalOd, "GFP": finalGfp}




def align_df(split_df, **kwargs):
    """
    This function takes a single experiment and aligns it to > stDev*5
    or you can pass in a kw od = num to set alignment
    """

    new_time = split_df["Time"].values
    st_dev =  np.std(split_df['OD'].iloc[0:10])
    od_filter_value = st_dev*5
    if kwargs:
        od_filter_value = kwargs.get('od')
    filtered_new = split_df.loc[split_df['OD'] > od_filter_value].reset_index(drop = True)
    filtered_new["Time"] = new_time[0:len(filtered_new)]

    return filtered_new

def generate_plots(long_df, directory):
    """
    This function takes in a long Df it should then output a graph for every
    group
    """

    split = long_df.groupby('Group')    
    split_df = [split.get_group(x) for x in split.groups]

    for num, df in enumerate(split_df):
        group_plot = (
            gg.ggplot(df)+
            gg.aes(x='Time', y='OD', color='variable')+
            gg.geom_point()+
            gg.ggtitle(df['Group'].values[0])
        )
        saveString = f"test{num}.png"
        gg.ggsave(group_plot, os.path.join(directory, saveString))


def calculate_regression(long_df):
    """
    This function will take the long_df and calculate a rolling window
    fit for growth rate
    """
    window_size = 8
    split = long_df.groupby('variable')    
    split_df = [split.get_group(x) for x in split.groups]
    long_df = []

    for df in split_df:
        regress_list = []
        for length in range(len(df)):
            res = stats.linregress(
                x = df['Time'].values[length:length+window_size],
                y = df['log(OD)'].values[length:length+window_size]
            )
            regress_list.append((res.slope*60))
        
        df['GrowthRate'] = regress_list
        long_df.append(df)
    
    long_df = pd.concat(long_df)
    return long_df

