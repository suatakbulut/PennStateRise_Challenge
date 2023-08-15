import ast
import os
import time
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def calculate_bmi_helper(weights, heights):
    """Calculates average bmi, ist std, and its trend's slop across the acquired dates

    Args:
        weights (_type_): a list of (date, weight) list
        heights (_type_): a list of (date, height) list

    Returns:
        list: average bmi, its std, as well as its trend
    """
    weights_dict = {date: weight for date, weight in weights}
    heights_dict = {date: height for date, height in heights}

    common_dates = set(weights_dict.keys()).intersection(
        set(heights_dict.keys()))

    bmis_dict = {}
    for date in common_dates:
        bmis_dict[date] = 703 * weights_dict[date] / (heights_dict[date]**2)

    sorted_dates = []
    sorted_bmis = []
    for date, bmi in sorted(bmis_dict.items(), key=lambda item: item[1]):
        sorted_dates.append(date)
        sorted_bmis.append(bmi)

    sorted_dates = np.array(sorted_dates)
    sorted_bmis = np.array(sorted_bmis)

    bmi_avg = sorted_bmis.mean()
    bmi_std = sorted_bmis.std()

    if sorted_bmis.size > 1:
        bmi_trend_slope = np.polyfit(sorted_dates, sorted_bmis, 1)[0]
    else:
        bmi_trend_slope = np.nan

    return [bmi_avg, bmi_std, bmi_trend_slope]


def calculate_bmi(row):
    """Returns bmi_avg, bmi_std, bmi_trend_slope for a given row

    Args:
        row (pd.DateFrame row / dictionary): 

    Returns:
        list: average bmi, its std, as well as its trend
    """
    weights = ast.literal_eval(row["Weight(lbs)"])
    heights = ast.literal_eval(row["Height(in)"])
    if not weights or not heights:
        return [np.nan, np.nan, np.nan]
    else:
        return calculate_bmi_helper(weights, heights)


def get_bmi(df):
    '''Augment the pd.DataFrame with 3 new columns: "bmi_avg", "bmi_std", "bmi_trend_slope"'''
    df[["bmi_avg", "bmi_std", "bmi_trend_slope"]] = pd.DataFrame(
        df.apply(calculate_bmi, axis=1).tolist(), index=df.index
    )
    return df


def clean_weight_helper(weights):
    """Calculates average weight, ist std, and its trend's slop across the acquired dates

    Args:
        weights (_type_): a list of (date, weight) list

    Returns:
        list: average weight, its std, as well as its trend
    """
    weights_dict = {date: weight for date, weight in weights}
    sorted_dates = []
    sorted_weights = []
    for date, weight in sorted(weights_dict.items(), key=lambda item: item[1]):
        sorted_dates.append(date)
        sorted_weights.append(weight)
    sorted_dates = np.array(sorted_dates)
    sorted_weights = np.array(sorted_weights)

    weight_avg = sorted_weights.mean()
    weight_std = sorted_weights.std()

    if sorted_weights.size > 1:
        weight_trend_slope = np.polyfit(sorted_dates, sorted_weights, 1)[0]
    else:
        weight_trend_slope = np.nan

    return [weight_avg, weight_std, weight_trend_slope]


def clean_weight(row):
    """Returns weight_avg, weight_std, weight_trend_slope for a given row

    Args:
        row (pd.DateFrame row / dictionary): 

    Returns:
        list: average weight, its std, as well as its trend
    """
    weights = ast.literal_eval(row["Weight(lbs)"])
    if not weights:
        return [np.nan, np.nan, np.nan]
    else:
        return clean_weight_helper(weights)


def get_weight(df):
    '''Augment the pd.DataFrame with 3 new columns: "weight_avg", "weight_std", "weight_trend_slope"'''
    df[["weight_avg", "weight_std", "weight_trend_slope"]] = pd.DataFrame(
        df.apply(clean_weight, axis=1).tolist(), index=df.index
    )
    return df


def get_date_columns():
    return [
        "smoking",
        "hypertension",
        "t1diabetes",
        "t2diabetes",
        "hypercholesterolemia",
        "hyperlipidemia",
        "ChronicObstructivePulmonaryDisease",
        "dialysis",
        "CongestiveHeartFailure",
        "ambulatory status",
        "anemia",
    ]


def get_date_value_columns():
    return [
        "aspirin",
        "P2Y12 antgaonist",
        "statin",
        "beta blocker",
        "AngiotensinConvertingEnzyme",
        "AngiotensinIIReceptorBlockers",
        "anticoagulant",
    ]


def date_column_helper(row, column):
    '''whether a feature is present (2), or not (1), or we don't have information on it(0)'''
    if row[column] == "0":
        return 0
    else:
        if row[column] == 0:
            return 1
        else:
            return 2


def clean_date_columns(df):
    '''For each column, determine whether a feature is present (2), or not (1), 
    or we don't have information on it(0)'''
    date_columns = get_date_columns()

    for col in date_columns:
        df[f"{col}_isPresent"] = df.apply(
            lambda row: date_column_helper(row, col), axis=1)

    df.drop(columns=date_columns, inplace=True)
    return df


def date_value_column_helper(row, column):
    '''For a given column in a row, expected to be a list of date value pairs, return the 
    daily total values, their std, and trend across time'''
    if row[column] == "0":
        return [np.nan, np.nan, np.nan]
    else:
        if row[column] == 0:
            return [0, 0, 0]
        else:
            date_value_pairs = ast.literal_eval(row[column])
            if isinstance(date_value_pairs[0], list):
                dates = np.array([date for date, value in date_value_pairs])
                values = [value.split()[0] for date, value in date_value_pairs]

            else:
                dates = np.array(date_value_pairs[::2])
                values = [val.split()[0] for val in date_value_pairs[1::2]]

            daily_total_values = defaultdict(float)
            for ind, (date, value) in enumerate(zip(dates, values)):
                if value != "Unknown":
                    daily_total_values[date] += float(value)

            sorted_dates = []
            sorted_values = []
            for date, value in sorted(daily_total_values.items(), key=lambda item: item[1]):
                sorted_dates.append(date)
                sorted_values.append(value)

            sorted_dates = np.array(sorted_dates)
            sorted_values = np.array(sorted_values)

            daily_total_value_avg = sorted_values.mean()
            daily_total_value_std = sorted_values.std()

            if sorted_values.size > 1:
                daily_total_value_trend_slope = np.polyfit(
                    sorted_dates, sorted_values, 1)[0]
            else:
                daily_total_value_trend_slope = np.nan

            return [daily_total_value_avg, daily_total_value_std, daily_total_value_trend_slope]


def clean_date_value_columns(df):
    '''For all the columns that are a list of date value pairs, create three new columsn: 
    daily total values, their std, and trend across time and drop the original one'''

    date_value_columns = get_date_value_columns()
    for col in date_value_columns:
        df[
            [
                f"{col}_daily_total_value_avg",
                f"{col}_daily_total_value_std",
                f"{col}_daily_total_value_trend_slope",
            ]
        ] = pd.DataFrame(
            df.apply(lambda row: date_value_column_helper(
                row, col), axis=1).tolist(),
            index=df.index,
        )
    # include "_isPresent" columns
    for col in date_value_columns:
        df[f"{col}_isPresent"] = df.apply(
            lambda row: date_column_helper(row, col), axis=1)

    df.drop(columns=date_value_columns, inplace=True)
    return df


def get_all_dates_info_helper(row):
    """Returns the number of unique days at which row has some information along with their range

    Args:
        row (pd.DataFrame row / dictionary):

    Returns:
        list: number of unique days for which the row has information, and their range
    """
    all_dates = np.array([])
    for col in get_date_value_columns():
        if row[col] not in ("0", 0):
            date_value_pairs = ast.literal_eval(row[col])
            if isinstance(date_value_pairs[0], list):
                dates = np.array([date for date, value in date_value_pairs])
            else:
                dates = np.array(date_value_pairs[::2])
            all_dates = np.append(all_dates, dates)

    for col in get_date_columns():
        if row[col] not in ("0", 0):
            date_value_pairs = ast.literal_eval(row[col])
            if isinstance(date_value_pairs, list):
                dates = date_value_pairs
                all_dates = np.append(all_dates, dates)

    dates_count = all_dates.size
    if dates_count:
        dates_unique = np.unique(all_dates)
        dates_nunique = dates_unique.size
        dates_range = dates_unique.max() - dates_unique.min()
        dates_freq = dates_nunique / (1+dates_range)
    else:
        dates_nunique = 0
        dates_range = np.nan
        dates_freq = np.nan

    # return dates_count, dates_nunique, dates_range, dates_freq
    return dates_nunique, dates_range


def get_all_dates_info(df):
    '''
    dates_nunique: For how many unique days the observation has data
    dates_range: The range of these days'''
    df[["dates_nunique", "dates_range"]] = pd.DataFrame(
        df.apply(lambda row: get_all_dates_info_helper(row), axis=1).tolist(),
        index=df.index,
    )
    return df


def get_dummies(df):
    '''Create female, black, and senior dummies since they are expected to be at high risk group'''
    df["is_female"] = pd.get_dummies(df["birth_gender"])["F"].astype(int)
    # one hot encode first for testing only
    df = pd.concat(
        [
            df,
            pd.get_dummies(df.ethnicity),
            pd.get_dummies(df.race),
        ],
        axis=1,
    )
    df["is_black"] = pd.get_dummies(
        df["race"])["Black or African American"].astype(int)
    df["is_senior"] = (df["age"] >= 75).astype(int)
    df.drop(columns=["ethnicity", "Unknown",
            "race", "birth_gender"], inplace=True)
    return df


def get_num_isPresents(df):
    '''Return the count of features that are present for an observation'''
    isPresent_cols = get_date_columns() + get_date_value_columns()
    all_isPresents = [df[f"{col}_isPresent"] == 2 for col in isPresent_cols]
    df["num_isPresents"] = sum(all_isPresents)
    return df


def include_interaction_terms(df):
    '''Create some interaction terms'''
    columns = get_date_columns() + get_date_value_columns()
    for col in columns:
        df[f"black_{col}_isPresent"] = df["is_black"] * df[f"{col}_isPresent"]
        df[f"female_{col}_isPresent"] = df["is_female"] * \
            df[f"{col}_isPresent"]
        df[f"senior_{col}_isPresent"] = df["is_senior"] * \
            df[f"{col}_isPresent"]

    for col in get_date_value_columns():
        df[f"black_{col}_daily_total_value_avg"] = df["is_black"] * \
            df[f"{col}_daily_total_value_avg"]
        df[f"female_{col}_daily_total_value_avg"] = df["is_female"] * \
            df[f"{col}_daily_total_value_avg"]
        df[f"senior_{col}_daily_total_value_avg"] = df["is_senior"] * \
            df[f"{col}_daily_total_value_avg"]

    return df


def get_original_data_path():
    return "../data/dataset.csv"


def get_processed_data_path():
    org_data_path = get_original_data_path()
    return org_data_path.replace("dataset.csv", "processed_dataset.pkl")


def prepare_data():
    '''Read the original data, clean its columns and create new features and return it'''
    data_path = get_original_data_path()
    df = pd.read_csv(data_path)

    df = get_bmi(df)
    df = get_weight(df)
    df = get_all_dates_info(df)
    df = get_dummies(df)

    # these two removes the original date related columns
    df = clean_date_columns(df)
    df = clean_date_value_columns(df)
    df = get_num_isPresents(df)

    # include interaction terms
    df = include_interaction_terms(df)

    return df
