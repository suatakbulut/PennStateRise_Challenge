import ast
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils_dataprep import *
from utils_modeling import *

warnings.filterwarnings("ignore")


def get_classifiers_and_grid():
    classifiers = {
        "log": LogisticRegression(),
        "rf": RandomForestClassifier(),
        "dt": DecisionTreeClassifier(),
        "nn": MLPClassifier(),
        "gnb": GaussianNB(),
        "quad": QuadraticDiscriminantAnalysis(),
    }

    grid_parameters = [
        [
            {
                "max_iter": [250, 370, 540],
                "penalty": ["l2", "none"],
            }
        ],
        [
            {
                "max_depth": [35, 25, 15, None],
                "n_estimators": [121, 151, 171, 251],
            }
        ],
        [
            {
                "max_depth": [20, 50, 100, None],
                "criterion": ["gini", "entropy", "log_loss"],
            }
        ],
        [
            {
                "hidden_layer_sizes": [[20, 20, 10], [20, 10], [20, 20]],
                "alpha": [0.0003, 0.03],
            }
        ],
        [
            {
                "var_smoothing": [1e-9],
            },
        ],
        [
            {
                "reg_param": [0.0],
            },
        ],
    ]
    return classifiers, grid_parameters


if __name__ == "__main__":
    df = obtain_df(use_saved_one=True)
    unwanted_column_endings = ["_std"]
    # unwanted_column_endings = []
    if unwanted_column_endings:
        m = "no_cols"
    else:
        m = "_".join(unwanted_column_endings)
    message = f"excluding_{m}"

    selected_columns = select_columns(
        df, unwanted_column_endings=unwanted_column_endings)

    X = df[selected_columns]
    y = df["Outcome 30 days Hospitalization"]

    print(f"X's shape: {X.shape} - y's shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    num_cols, cat_cols = categorize_columns(X_train)

    preprocessor = get_preprocessor(num_cols, cat_cols)

    print(
        f"\nThere are {len(cat_cols)} categorical columns and {len(num_cols)} numerical columns.\n")
    print(
        f"Before preprocessing.. X_train: {X_train.shape} - X_test: {X_test.shape}")
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    print(
        f"After preprocessing..  X_train: {X_train.shape} - X_test: {X_test.shape}")

    clf1 = LogisticRegression()
    single_model_results(clf1, X_train, X_test, y_train, y_test)

    clf2 = RandomForestClassifier()
    single_model_results(clf2, X_train, X_test, y_train, y_test)

    classifiers, grid_parameters = get_classifiers_and_grid()
    results_df = train_multiple_gridsearch(classifiers, grid_parameters, X_train, X_test, y_train, y_test, message=message)
