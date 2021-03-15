import os
import pandas as pd
import numpy as np
import csv
from scipy import stats


def analyze_plate(filepath, alignment_value):

    plate_normalized = normalize_plate(filepath)

    time = 'Time (min)'
    # Align the dataframe to a specific value for individual values

    normalized_od_df = plate_normalized['OD'].copy()
    final_od_normalized = normalized_od_df.astype('float')
    final_od_normalized = final_od_normalized.melt(id_vars=time)
    final_od_normalized = final_od_normalized.rename(columns=dict(
        zip(final_od_normalized.columns, ["Time", "variable", "norm_OD"])))

    od_df = plate_normalized['OD'].melt(id_vars=time)
    od_df = od_df.rename(columns=dict(
        zip(od_df.columns, ["Time", "variable", "OD"])))
    od_df.loc[:, 'norm_OD'] = final_od_normalized["norm_OD"]

    gfp_df = plate_normalized['GFP'].melt(id_vars=time)
    gfp_df = gfp_df.rename(columns=dict(
        zip(gfp_df.columns, ["Time", "variable", "GFP"])))

    merged = od_df.merge(gfp_df)

    osmolarity = plate_normalized['osmolarity'].set_index('Group')
    osmolarity_dict = osmolarity.to_dict()

    merged.loc[:, 'osmolarity'] = merged['variable'].str[3:7].astype(float)
    merged.loc[:, 'osmolarity'] = merged['osmolarity'].map(
        osmolarity_dict['osmolarity'])
    # Here I add in all the osmolarity values extracted from the excel

    # OK its in the long format now to align it to OD

    aligned_df_long = merged.groupby('variable').apply(
        align_df, align_limit = alignment_value).reset_index(drop=True)

    aligned_df_long.loc[:, 'Group'] = aligned_df_long['variable'].apply(
        lambda x: x[0:7])
    aligned_df_long.loc[:, 'GFP/OD'] = aligned_df_long['GFP'] / \
        aligned_df_long['norm_OD']
    aligned_df_long.loc[:, 'log(OD)'] = np.log(aligned_df_long['OD'])

    aligned_df_long = calculate_regression(aligned_df_long)

    return aligned_df_long


def align_df(split_df, align_limit, **kwargs):
    """
    This function takes a single experiment and aligns it to > stDev*5
    or you can pass in a kw od = num to set alignment
    """
    alignment_value = align_limit

    new_time = split_df["Time"].values
    st_dev = np.std(split_df['OD'].iloc[0:10])
    mean = np.mean(split_df['OD'].iloc[0:10])
    od_filter_value = alignment_value
    if kwargs:
        od_filter_value = kwargs.get('od')

    filtered_new = split_df.loc[split_df['OD'] >
                                od_filter_value].reset_index(drop=True)
    filtered_new.loc[:, "Time"] = new_time[0:len(filtered_new)]

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
            regress_list.append(res.slope)

        df.loc[:, 'GrowthRate'] = pd.Series(regress_list, index=df.index)
        reconstructed_df.append(df)

    growth_rate_calculated_df = pd.concat(
        reconstructed_df).reset_index(drop=True)

    return growth_rate_calculated_df


def normalize_plate(path_to_excel):
    """
    This function takes in the path_to_excel file and outputs the normalized GFP and OD values as a dict
    """
    # so first we want to open and analyse a single file

    input_df = pd.read_excel(path_to_excel)

    # columns k and l add in plate osmo values
    osmolarity_values = input_df.iloc[0:6, 10:12]

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
    input_df_od.loc[:, "Time (min)"] = new_time
    input_df_gfp.loc[:, "Time (min)"] = new_time

    # now I want to subtract the average of the first row from all values in the dataframe
    # turns out not subtracting the baseline works way betetr when calculating the rolling growth rate
    # this is because small log OD values throw of the calculation

    final_od = input_df_od
    # final_od_raw = final_od.astype('float')

    first_row_od = input_df_od.iloc[0:100, 1:].min(axis=0)
    final_od.iloc[:, 1:] = input_df_od.iloc[:, 1:].apply(
        lambda row: row - first_row_od, axis=1)
    final_od = final_od.astype('float')

    final_gfp = input_df_gfp
    #final_gfp = final_gfp.astype('float')

    first_row_gfp = input_df_gfp.iloc[0:100, 1:].min(axis=0)
    final_gfp.iloc[:, 1:] = input_df_gfp.iloc[:, 1:].apply(
        lambda row: row - first_row_gfp, axis=1)
    final_gfp = final_gfp.astype('float')

    return {"OD": final_od, "GFP": final_gfp, "osmolarity": osmolarity_values}
