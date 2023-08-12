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

from utils_data import *
from utils_model import *

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    unwanted_column_endings = ["_std", "race"]
    # unwanted_column_endings = []
    use_saved_one = False 
    X_train, X_test, y_train, y_test = split_and_preprocess_data(
        use_saved_one=use_saved_one, unwanted_column_endings=unwanted_column_endings, test_size=0.30)

    clf1 = LogisticRegression()
    single_model_results(clf1, X_train, X_test, y_train, y_test)

    clf2 = RandomForestClassifier()
    single_model_results(clf2, X_train, X_test, y_train, y_test)

    if unwanted_column_endings:
        m = "no_cols"
    else:
        m = "_".join(unwanted_column_endings)
    message = f"excluding_{m}"

    # classifiers, grid_parameters = get_classifiers_and_grid()
    # results_df = train_multiple_gridsearch(
    #     classifiers, grid_parameters, X_train, X_test, y_train, y_test, message=message)
