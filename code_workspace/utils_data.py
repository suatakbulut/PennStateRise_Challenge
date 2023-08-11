import ast
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

def calculate_bmi(row):
    """bmi_avg, bmi_std, bmi_trend_slope"""
    weights = ast.literal_eval(row["Weight(lbs)"])
    heights = ast.literal_eval(row["Height(in)"])
    if not weights or not heights:
        return [np.nan, np.nan, np.nan]
    else:
        bmis = []
        for [tw, weight], [th, height] in zip(weights, heights):
            if tw == th:
                bmi = 703 * weight / (height**2)
                bmis.append(bmi)
        bmis = np.array(bmis)
        bmi_avg = bmis.mean()
        bmi_std = bmis.std()
        if bmis.size > 1:
            bmi_trend_slope = np.polyfit(np.arange(bmis.size), bmis, 1)[0]
        else:
            bmi_trend_slope = 0

        return [bmi_avg, bmi_std, bmi_trend_slope]


def obtain_bmi(df):
    df[["bmi_avg", "bmi_std", "bmi_trend_slope"]] = pd.DataFrame(
        df.apply(calculate_bmi, axis=1).tolist(), index=df.index
    )
    return df


def date_column_helper(row, column):
    """isPresent, Frequency, datediff mean, datediff std, datediff_trendline_slope"""
    if row[column] == "0":
        return [np.nan, np.nan, np.nan, np.nan, np.nan]
    else:
        #         answer_dates = ast.literal_eval(row[column])
        if row[column] == 0:
            return [0, np.nan, np.nan, np.nan, np.nan]
        else:
            dates = np.array(ast.literal_eval(row[column]))
            freq = dates.size / (1 + dates.max() - dates.min())
            datediff = dates[1:] - dates[:-1]
            if datediff.size > 1:
                trend_slope = np.polyfit(np.arange(datediff.size), datediff, 1)[0]
            else:
                trend_slope = 0
            return [1, freq, datediff.mean(), datediff.std(), trend_slope]


def date_value_column_helper(row, column):
    """isPresent, Frequency, datediff mean, datediff std, datediff_trendline_slope, dosage mean, dosage std, trend_line_slope"""
    if row[column] == "0":
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    else:
        #         answer_dates = ast.literal_eval(row[column])
        if row[column] == 0:
            return [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        else:
            date_value_pairs = ast.literal_eval(row[column])
            if isinstance(date_value_pairs[0], list):
                dates = np.array([date for date, value in date_value_pairs])
                values = [value.split()[0] for date, value in date_value_pairs]

            else:
                dates = np.array(date_value_pairs[::2])
                values = date_value_pairs[1::2]
                values = [val.split()[0] for val in values]

            # calculate datediff related parameters
            datediff = dates[1:] - dates[:-1]
            freq = dates.size / (1 + dates.max() - dates.min())
            datediff_mean = datediff.mean()
            datediff_std = datediff.std()
            if datediff.size > 1:
                datediff_trend_slope = np.polyfit(
                    np.arange(datediff.size), datediff, 1
                )[0]
            else:
                datediff_trend_slope = 0

            # calculate value related parameters
            new_values = []
            new_dates = []
            for date, value in zip(dates, values):
                if value != "Unknown":
                    new_values.append(float(value))
                    new_dates.append(date)

            new_values = np.array(new_values)
            new_dates = np.array(new_dates)
            if new_values.size == 0:
                values_mean = np.nan
                values_std = np.nan
                values_trend_slope = np.nan

            else:
                values_mean = new_values.mean()
                values_std = new_values.std()
                if new_values.size > 1:
                    values_trend_slope = np.polyfit(new_dates, new_values, 1)[0]
                else:
                    values_trend_slope = 0

            return [
                1,
                freq,
                datediff_mean,
                datediff_std,
                datediff_trend_slope,
                values_mean,
                values_std,
                values_trend_slope,
            ]


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


def clean_date_columns(df):
    """11 columns"""
    date_columns = get_date_columns()
    for col in date_columns:
        df[
            [
                f"{col}_isPresent",
                f"{col}_freq",
                f"{col}_datediff_avg",
                f"{col}_datediff_std",
                f"{col}_datediff_trend_slope",
            ]
        ] = pd.DataFrame(
            df.apply(lambda row: date_column_helper(row, col), axis=1).tolist(),
            index=df.index,
        ).astype(
            "float"
        )

    df.drop(columns=date_columns, inplace=True)
    return df


def clean_date_value_columns(df):
    """7 columns"""
    date_value_columns = get_date_value_columns()
    for col in date_value_columns:
        df[
            [
                f"{col}_isPresent",
                f"{col}_freq",
                f"{col}_datediff_avg",
                f"{col}_datediff_std",
                f"{col}_datediff_trend_slope",
                f"{col}_values_avg",
                f"{col}_values_std",
                f"{col}_values_trend_slope",
            ]
        ] = pd.DataFrame(
            df.apply(lambda row: date_value_column_helper(row, col), axis=1).tolist(),
            index=df.index,
        ).astype(
            "float"
        )
    df.drop(columns=date_value_columns, inplace=True)
    return df


def get_gender_dummies(df):
    df["birth_gender"] = pd.get_dummies(df["birth_gender"])["F"]
    return df


def include_interaction_terms(df):
    df["is_black"] = df["race"] == "Black or African American"
    df["gender_bmi"] = df["birth_gender"] * df["bmi_avg"]
    df["gender_age"] = df["birth_gender"] * df["age"]
    df["gender_black"] = df["birth_gender"] * df["is_black"]

    date_value_columns = get_date_value_columns()
    for col in date_value_columns:
        df[f"black_{col}"] = df["is_black"] * df[f"{col}_isPresent"]
        df[f"gender_{col}"] = df["birth_gender"] * df[f"{col}_isPresent"]

    date_columns = get_date_columns()
    for col in date_columns:
        df[f"black_{col}"] = df["is_black"] * df[f"{col}_isPresent"]
        df[f"gender_{col}"] = df["birth_gender"] * df[f"{col}_isPresent"]

    # df.drop(columns=["is_black"], inplace=True) ## this jumps the performance of the model
    return df


def get_original_data_path():
    return "../data/dataset.csv"


def get_processed_data_path():
    org_data_path = get_original_data_path()
    path_parts = org_data_path.split("/")
    file_name = "_".join(["processed", path_parts[-1]])
    directory = "/".join(path_parts[:-1])
    processed_data_path = "/".join([directory, file_name])
    return processed_data_path


def prepare_data():
    data_path = get_original_data_path()
    df = pd.read_csv(data_path)

    df.labTest = df.labTest.astype(object)
    df.PAD = df.PAD.astype(object)

    df = obtain_bmi(df)
    df = clean_date_columns(df)
    df = clean_date_value_columns(df)

    # include interaction terms
    df = get_gender_dummies(df)
    df = include_interaction_terms(df)
    return df


def obtain_df(use_saved_one=True):
    print("\n\n\n")
    processed_data_path = get_processed_data_path()
    processed_file_path = processed_data_path.split("/")[-1]

    if use_saved_one:
        if processed_file_path in os.listdir("../data"):
            print(f"Loading processed data from {processed_data_path}..")
            df = pd.read_csv(processed_data_path)
        else:
            print("Processed data does not exist. Creating one.")
            df = prepare_data()
            print(f"Processed data is created and saved at {processed_data_path}..")
            df.to_csv(processed_data_path)
    else:
        df = prepare_data()
        print(f"Processed data is created and saved at {processed_data_path}..")
        df.to_csv(processed_data_path)

    return df

