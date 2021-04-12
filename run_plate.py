import os
import pandas as pd
import numpy as np
import csv
from scipy import stats


def subtract_background(df, modifier):
    first_row = df.iloc[0:10, 1:].mean(axis=0).copy()
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(
        lambda row: row - (first_row - modifier), axis=1)
    df = df.astype('float').copy()
    return df


def convert_time_to_hours(df):
    df = df.rename(columns={"Time (min)": "Time"})
    new_time = pd.Series(df["Time"] / 3600).astype(float).round(3)
    df["Time"] = new_time
    return df


def load_plate(path_to_excel, label):
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
    if label == 'standard':
        with open('Data/01_helper_data/platereaderLayout.csv', newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
    elif label == '100-600':
        with open('Data/01_helper_data/platereaderLayout100to600.csv', newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

    col_names = data[0]

    input_df_od.rename(columns=dict(
        zip(input_df_od.columns, col_names)),  inplace=True)
    input_df_gfp.rename(columns=dict(
        zip(input_df_gfp.columns, col_names)),  inplace=True)

    final_od = convert_time_to_hours(input_df_od)
    final_gfp = convert_time_to_hours(input_df_gfp)
    final_od = subtract_background(final_od, 0.005)
    final_gfp = subtract_background(final_gfp, 0)

    return {"OD": final_od, "GFP": final_gfp, "osmolarity": osmolarity_values}


def analyze_plate(filepath, alignment_value, label):

    plate_normalized = load_plate(filepath, label)

    time = 'Time'
    # Align the dataframe to a specific value for individual values

    od_df = plate_normalized['OD'].melt(id_vars=time)
    od_df = od_df.rename(columns=dict(
        zip(od_df.columns, ["Time", "variable", "OD"])))

    gfp_df = plate_normalized['GFP'].melt(id_vars=time)
    gfp_df = gfp_df.rename(columns=dict(
        zip(gfp_df.columns, ["Time", "variable", "GFP"])))

    merged = od_df.merge(gfp_df)

    osmolarity = plate_normalized['osmolarity'].set_index('Group')
    osmolarity_dict = osmolarity.to_dict()

    merged['osmolarity'] = merged['variable'].str[3:7].astype(float)
    merged['osmolarity'] = merged['osmolarity'].map(
        osmolarity_dict['osmolarity'])
    # Here I add in all the osmolarity values extracted from the excel

    merged['Group'] = merged['variable'].str[0:7]
    merged['GFP/OD'] = merged['GFP'] / merged['OD']
    merged['log(OD)'] = np.log(merged['OD'] / np.mean(merged['OD'].values[0:10]))

    analyzed_plate = merged

    return analyzed_plate


def align_df(df, align_limit, **kwargs):
        """
        This function takes a single experiment and aligns it
        """
        rebuilt_df = []

        for name, df in df.groupby('variable'):

            new_time = df["Time"].values
            st_dev = np.std(df['OD'].iloc[0:10])
            mean = np.mean(df['OD'].iloc[0:10])
            
            #od_filter_value = (mean+(st_dev*5))
            od_filter_value = align_limit

            filtered_new = df[df['OD'] > od_filter_value].reset_index(drop=True).copy()
            filtered_new["Time"] = new_time[0:len(filtered_new)]
            rebuilt_df.append(filtered_new)
        
        aligned_df = pd.concat(rebuilt_df).reset_index(drop=True)
   
        return aligned_df