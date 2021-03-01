import os
import pandas as pd
import numpy as np
import csv
import plotnine as gg
from scipy import stats


def analyze_plate(filepath):


    plate_normalized = normalize_plate(filepath)

    time = 'Time (min)'
    # Align the dataframe to a specific value for individual values

    od_df = plate_normalized['OD'].melt(id_vars=time)
    od_df = od_df.rename(columns=dict(zip(od_df.columns, ["Time", "variable", "OD"])))

    gfp_df = plate_normalized['GFP'].melt(id_vars=time)
    gfp_df = gfp_df.rename(columns=dict(zip(gfp_df.columns, ["Time", "variable", "GFP"])))

    merged = od_df.merge(gfp_df)

    osmolarity = plate_normalized['osmolarity']
    print(osmolarity)

    merged['osmolarity'] = int(merged['variable'].str[3:7])

    # osmo_dict = {osmolarity}
    # print(osmo_dict)
    # merged.replace({"osmolarity": osmo_dict})
    print(merged)



    # OK its in the long format now to align it to OD

    aligned_df_long = merged.groupby('variable').apply(
        align_df).reset_index(drop=True)

    aligned_df_long['Group'] = aligned_df_long['variable'].apply(
        lambda x: x[0:7])
    aligned_df_long['GFP/OD'] = aligned_df_long['GFP'] / aligned_df_long['OD']
    aligned_df_long['log(OD)'] = np.log(aligned_df_long['OD'])

    aligned_df_long = calculate_regression(aligned_df_long)

    return aligned_df_long


def align_df(split_df, **kwargs):
    """
    This function takes a single experiment and aligns it to > stDev*5
    or you can pass in a kw od = num to set alignment
    """

    new_time = split_df["Time"].values
    st_dev = np.std(split_df['OD'].iloc[0:10])
    od_filter_value = st_dev*10
    if kwargs:
        od_filter_value = kwargs.get('od')
    filtered_new = split_df.loc[split_df['OD'] >
                                od_filter_value].reset_index(drop=True)
    filtered_new["Time"] = new_time[0:len(filtered_new)]

    return filtered_new


def calculate_regression(df, window_size=8):

    split = df.groupby('variable')
    split_df = [split.get_group(x) for x in split.groups]

    reconstructed_df = []

    for df in split_df:
        regress_list = []
        for length in range(len(df)):
            res = stats.linregress(
                x=df['Time'].values[length:length+window_size],
                y=df['log(OD)'].values[length:length+window_size]
            )
            regress_list.append((res.slope))

        df['GrowthRate'] = regress_list
        reconstructed_df.append(df)
    
    growth_rate_calculated_df = pd.concat(reconstructed_df).reset_index(drop=True)

    return growth_rate_calculated_df



def normalize_plate(path_to_excel):
    """
    This function takes in the path_to_excel file and outputs the normalized GFP and OD values as a dict
    """
    # so first we want to open and analyse a single file

    input_df = pd.read_excel(path_to_excel)

    osmolarity_values = input_df.iloc[0:6,10:12] #columns k and l add in plate osmo values

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

    # now drop the first row as it contains our headers and drop any NA from empty data
    input_df_od = input_df_od.drop(input_df_od.index[0]).dropna()
    input_df_gfp = input_df_gfp.drop(input_df_gfp.index[0]).dropna()

    # Now we need to name the columns correctly
    # Here I call the column names saved in a csv and create a list from them named colNames

    with open('Data/01_helper_data/platereaderLayout.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    col_names = data[0]

    input_df_od.rename(columns=dict(
        zip(input_df_od.columns, col_names)),  inplace=True)
    input_df_gfp.rename(columns=dict(
        zip(input_df_gfp.columns, col_names)),  inplace=True)

    # now find the Time in minutes
    new_time = [round((x*(7.6)/60), 2) for x in range(0, len(input_df_od))]
    input_df_od["Time (min)"] = new_time
    input_df_gfp["Time (min)"] = new_time

    # now I want to subtract the average of the first row from all values in the dataframe

    first_row_od = input_df_od.iloc[[0]].values[0]
    final_od = input_df_od.apply(lambda row: row - first_row_od, axis=1)

    first_row_gfp = input_df_gfp.iloc[[0]].values[0]
    final_gfp = input_df_gfp.apply(lambda row: row - first_row_gfp, axis=1)

    return {"OD": final_od, "GFP": final_gfp, "osmolarity": osmolarity_values}


