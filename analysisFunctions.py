import os
import pandas as pd
import numpy as np
import csv
import plotnine as gg
from scipy import stats


def analyze_plate(file, **kwargs):
    keywords = kwargs

    root = file['root']
    filename = file['filename']

    plate_path = os.path.join(root, filename)
    plate_normalized = normalize_plate(plate_path)

    # Align the dataframe to a specific value for individual values

    a = plate_normalized['OD'].melt(id_vars="Time (min)")
    a = a.rename(columns=dict(zip(a.columns, ["Time", "variable", "OD"])))

    b = plate_normalized['GFP'].melt(id_vars="Time (min)")
    b = b.rename(columns=dict(zip(b.columns, ["Time", "variable", "GFP"])))

    merged = a.merge(b)

    # OK its in the long format now to align it to OD

    aligned_df_long = merged.groupby('variable').apply(
        align_df).reset_index(drop=True)

    aligned_df_long['Group'] = aligned_df_long['variable'].apply(
        lambda x: x[0:7])
    aligned_df_long['GFP/OD'] = aligned_df_long['GFP'] / aligned_df_long['OD']
    aligned_df_long['log(OD)'] = np.log(aligned_df_long['OD'])

    if keywords.get('plot'):
        print("generating plots")
        generate_plots(aligned_df_long, root)

    aligned_df_long = calculate_regression(aligned_df_long)

    return aligned_df_long


def align_df(split_df, **kwargs):
    """
    This function takes a single experiment and aligns it to > stDev*5
    or you can pass in a kw od = num to set alignment
    """

    new_time = split_df["Time"].values
    st_dev = np.std(split_df['OD'].iloc[0:10])
    od_filter_value = st_dev*5
    if kwargs:
        od_filter_value = kwargs.get('od')
    filtered_new = split_df.loc[split_df['OD'] >
                                od_filter_value].reset_index(drop=True)
    filtered_new["Time"] = new_time[0:len(filtered_new)]

    return filtered_new


def calculate_regression(df, window_size=8):
    regress_list = []
    for length in range(len(df)):
        res = stats.linregress(
            x=df['Time'].values[length:length+window_size],
            y=df['log(OD)'].values[length:length+window_size]
        )
        regress_list.append((res.slope))

    df['GrowthRate'] = regress_list
    return df


def calculate_regression_long(long_df):
    """
    This function will take the long_df and calculate a rolling window
    fit for growth rate
    """
    long_df = long_df.groupby('variable').apply(
        calculate_regression).reset_index(drop=True)
    return long_df


def calculate_max_growth_rate(df):
    df = df[df['OD'] > 0.02]
    split = df.groupby('variable')
    split_df = [split.get_group(x) for x in split.groups]
    max_growth_rate_list = []

    for df in split_df:
        max_growth_rate = max(df['GrowthRate'])
        max_growth_rate_list.append(max_growth_rate)

    return max_growth_rate_list


def generate_plots(long_df, directory):
    """
    This function takes in a long Df it should then output a graph for every
    group
    """

    split = long_df.groupby('Group')
    split_df = [split.get_group(x) for x in split.groups]

    for num, df in enumerate(split_df):
        group_plot = (
            gg.ggplot(df) +
            gg.aes(x='Time', y='OD', color='variable') +
            gg.geom_point() +
            gg.ggtitle(df['Group'].values[0])
        )
        save_string = f"test{num}.png"
        gg.ggsave(group_plot, os.path.join(directory, save_string))


def normalize_plate(path_to_excel):
    """
    This function takes in the path_to_excel file and outputs the normalized GFP and OD values as a dict
    """
    # so first we want to open and analyse a single file

    input_df = pd.read_excel(path_to_excel)

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

    return {"OD": final_od, "GFP": final_gfp}


# for fitderiv here

# def calculate_growth_rate(df, OD_filter=0.02):
#     sys.stdout = open(os.devnull, 'w')  # block fitderiv from printing
#     split = test.groupby('variable')
#     split_df = [split.get_group(x) for x in split.groups]

#     growth_rate_list = []

#     for df in split_df:
#         OD = df['OD'].values[df['OD'].values > OD_filter]
#         time = df['Time'].values[df['OD'].values > OD_filter]

#         q = fd.fitderiv(time, OD)
#         temp_gr = {"Sample": df['variable'].values[0],
#                    "Growth_Rate": q.ds.get('max df')}
#         growth_rate_list.append(temp_gr)

#     sys.stdout = sys.__stdout__  # re enable print

#     return growth_rate_list


# def longest_run(myList):
#     prev = 0
#     size = 0
#     max_size = 0
#     max_list = []

#     for num, i in enumerate(myList):
#         if i > (prev *0.99) and i < (prev * 1.01):
#             size += 1
#             if size > max_size:
#                 max_size = size
#                 max_list.append(num-1)
#         else:
#             size = 0
#         prev = i
#     return max_size+1, (max(max_list)-max_size+1)
