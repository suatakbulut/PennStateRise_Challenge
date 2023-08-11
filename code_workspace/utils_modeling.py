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

warnings.filterwarnings("ignore")

def get_preprocessor(num_cols, cat_cols):
    # Preprocessing Numerical data
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            # ("poly", PolynomialFeatures())
        ]
    )

    # Preprocessing Categorical data
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            # ("interaction", PolynomialFeatures(interaction_only=True))
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    return preprocessor


def single_model_results(clf, X_train, X_test, y_train, y_test):
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = (time.time() - start) / 60

    print(f"\nTraining with {clf} took {train_time:.2f} mins.")

    # Calculate metrics
    clf_probs = clf.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, clf_probs[:, 1])
    y_pred = clf.predict(X_test)
    f_score = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"F1 Score  : {f_score:.4f}")
    print(f"AUC Score : {auc_score:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}\n")


def select_columns(df, unwanted_column_endings=[]):

    unwanted_columns = [
        "patient_id",
        "Outcome 30 days Hospitalization",
        "Height(in)",
        "Weight(lbs)",
    ]

    for col in df.columns:
        for unwanted_ending in unwanted_column_endings:
            if col.endswith(unwanted_ending):
                unwanted_columns.append(col)
                break
    print("Excluding the following columns..")
    print(unwanted_columns)
    selected_columns = sorted(list(set(df.columns).difference(set(unwanted_columns))))
    return selected_columns


def categorize_columns(X):
    num_cols = [col for col in X.columns if X[col].dtype in ["int64", "float"]]
    cat_cols = [col for col in X.columns if X[col].dtype == "object"]
    return num_cols, cat_cols


def split_and_preprocess_data(df, unwanted_column_endings=[], test_size=0.30 ):
    selected_columns = select_columns(df, unwanted_column_endings=unwanted_column_endings)

    X = df[selected_columns]
    y = df["Outcome 30 days Hospitalization"]

    print(f"X's shape: {X.shape} - y's shape: {y.shape}") 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    num_cols, cat_cols = categorize_columns(X_train)
    preprocessor = get_preprocessor(num_cols, cat_cols)

    print(f"\nThere are {len(cat_cols)} categorical columns and {len(num_cols)} numerical columns.\n")
    print(f"Before preprocessing.. X_train: {X_train.shape} - X_test: {X_test.shape}")
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    print(f"After preprocessing..  X_train: {X_train.shape} - X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_single_gridsearch(classifier, grid_parameter, X_train, X_test, y_train, y_test):
    # run grid search to find best params
    scorer = make_scorer(recall_score)
    clf = GridSearchCV(classifier, grid_parameter, cv=5, scoring=scorer, n_jobs=-1)

    # fit Logistic Regression
    classifier_name = str(classifier).replace("()", "")
    title = f"\n\nTraining a GridSearchCV model with \n{classifier_name} predictor.."
    print(title)
    print(len(title) * "=")
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = round((time.time() - start) / 60, 2)
    print(f"Fitting took {train_time} mins.")

    # Generate predicted probabilites
    start = time.time()
    clf_probs = clf.predict_proba(X_test)
    predict_time = round((time.time() - start) / 60, 2)
    print(f"Prediction took {predict_time} mins.")

    # Calculate metrics
    auc_score = roc_auc_score(y_test, clf_probs[:, 1])
    y_pred = clf.predict(X_test)
    f_score = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"F1 Score  : {f_score:.4f}")
    print(f"AUC Score : {auc_score:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}\n\n")

    return [
        classifier_name,
        clf,
        train_time,
        predict_time,
        f_score,
        auc_score,
        accuracy,
        precision,
        recall,
    ]


def train_multiple_gridsearch(classifiers, grid_parameters, X_train, X_test, y_train, y_test, message=""):
    results_df = pd.DataFrame(
        columns=[
            "classifier_class",
            "trained_model",
            "train_time",
            "predict_time",
            "F1_Score",
            "AUC",
            "Accuracy",
            "Precision",
            "Recall",
        ]
    )

    for [test, classifier], grid_parameter in zip(classifiers.items(), grid_parameters):
        [
            classifier_class,
            trained_model,
            train_time,
            predict_time,
            f_score,
            auc_score,
            accuracy,
            precision,
            recall,
        ] = train_single_gridsearch(classifier, grid_parameter, X_train, X_test, y_train, y_test)

        results_df.loc[len(results_df)] = [
            classifier_class,
            trained_model,
            train_time,
            predict_time,
            f_score,
            auc_score,
            accuracy,
            precision,
            recall,
        ]

    results_df = results_df.sort_values(by="Recall", ascending=False)
    
    best_accuracy = results_df.Accuracy.values[0]
    best_recall = results_df.Recall.values[0]
    output_path = f"../trained_models/results_df_{best_accuracy:.3f}_Acc_{best_recall:.3f}_Rec_{message}.pkl"
    print(f"Pickling trained models dataframe at {output_path}")
    results_df.to_pickle(output_path)
    
    # print the results to the screen
    pd.set_option('display.width', 1000) 
    pd.set_option('display.max_columns', 20)
    print(results_df)
    
    return results_df

