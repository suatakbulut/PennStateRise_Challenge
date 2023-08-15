import time

import numpy as np
import pandas as pd
from sklearn.compose import (ColumnTransformer, make_column_selector,
                             make_column_transformer)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder,
                                   PolynomialFeatures, StandardScaler)
from sklearn.tree import DecisionTreeClassifier


def get_preprocessor():
    """Cretes and Returns a preprocessor that handles categorical and numeric columns only"""
    # Even though there is no categorical column, putting one here
    cat_selector = make_column_selector(dtype_include=object)
    num_selector = make_column_selector(dtype_include=np.number)

    cat_processor = OneHotEncoder(handle_unknown="ignore", drop="if_binary")
    num_processor = make_pipeline(
        SimpleImputer(strategy="constant"),
        StandardScaler(),
    )

    preprocessor = make_column_transformer(
        (num_processor, num_selector),
        (cat_processor, cat_selector),
    )

    return preprocessor


def get_predictor(pred, params=None):
    '''Returns a sklearn predictor. If params is specified, then set the corresponding 
    params in teh predictor first.'''
    predictors = _predictor_dict()
    if pred in predictors:
        predictor = predictors[pred]
        if params:
            predictor.set_params(**params)
        return predictor
    else:
        raise ValueError(
            f"Available predictors are {list(predictors.keys())}..")


def _predictor_dict():
    '''Considering only these predictors'''
    predictors = {
        "logistic": LogisticRegression(solver="sag", max_iter=500),
        "forest": RandomForestClassifier(),
        "xgb": GradientBoostingClassifier(),
        "gaussian": GaussianNB(),
        "quad": QuadraticDiscriminantAnalysis(),
        "decision_tree": DecisionTreeClassifier(),
    }
    return predictors


def create_estimator_with(pred, params=None):
    '''Bundle a predictor with a fresh preprocessor to create an estimator and avoid leakage'''
    preprocessor = get_preprocessor()
    predictor = get_predictor(pred, params=params)
    estimator = Pipeline(
        steps=[("preprocess", preprocessor), ("predictor", predictor)])
    return estimator


def get_single_model_params(predictor):
    '''When calling a single model only case, use the following parameters'''
    if predictor in ("forest", "decision_tree"):
        params = {
            "max_depth": 25,
            "n_estimators": 65,
            "max_features": "sqrt",
            "min_samples_split": 16,
            "criterion": "gini",
        }
    elif predictor == "xgb":
        params = {
            "max_features": "sqrt",
            "min_samples_split": 16,
        }
    else:
        params = None
    return params


def single_model_results(predictor, X_train, X_test, y_train, y_test):
    """Creates an estimator with a given predictor, trains it, and returns 
    its performance on both train and test sets

    Args:
        predictor (string): Has to be in the keys of _predictor_dict()
        X_train (pd.DataFrame): train set
        X_test (pd.DataFrame): test set
        y_train (pd.Series): train labels
        y_test (pd.Series): test labels
    """
    params = get_single_model_params(predictor)
    clf = create_estimator_with(predictor, params=params)
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = (time.time() - start) / 60
    classifier_name = str(clf.named_steps["predictor"]).split('(')[0]
    print(f"\nTraining with {classifier_name} took {train_time:.2f} mins.")

    # Calculate metrics
    auc_score, f_score, accuracy, precision, recall = calculate_metrics(
        clf, X_test, y_test)
    auc_score_tr, f_score_tr, accuracy_tr, precision_tr, recall_tr = calculate_metrics(
        clf, X_train, y_train)

    print("===========================")
    print(f"{classifier_name}".center(27, " "))
    print("===========================")
    print("== Test Set vs Train Set ==")
    print(f"F1 Score  : {f_score:.4f} | {f_score_tr:.4f}")
    print(f"AUC Score : {auc_score:.4f} | {auc_score_tr:.4f}")
    print(f"Accuracy  : {accuracy:.4f} | {accuracy_tr:.4f}")
    print(f"Precision : {precision:.4f} | {precision_tr:.4f}")
    print(f"Recall    : {recall:.4f} | {recall_tr:.4f}")
    print("===========================\n")


def select_columns(df, unwanted_column_endings=[]):
    '''Excludes some columsn from our features. Exludes columns if they end 
    with the suffix specified in unwanted_column_endings'''
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
    selected_columns = sorted(
        list(set(df.columns).difference(set(unwanted_columns))))
    return selected_columns


def calculate_metrics(clf, X_test, y_test):
    clf_probs = clf.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, clf_probs[:, 1])
    y_pred = clf.predict(X_test)
    f_score = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return auc_score, f_score, accuracy, precision, recall


def train_single_gridsearch(predictor, grid_parameter, X_train, X_test, y_train, y_test):
    """Train a gridsearch cv model by creating an estimator with the specified predcitor. 
    Return the precitor's name, trained model, training time, and its performance

    Args:
        predictor (string): Has to be in the keys of _predictor_dict()
        grid_parameter (_type_): grid on which to optimize the estimator
        X_train (pd.DataFrame): train set
        X_test (pd.DataFrame): test set
        y_train (pd.Series): train labels
        y_test (pd.Series): test labels

    Returns:
        list: _description_
    """
    # run grid search to find best params
    # scorer = make_scorer(roc_auc_score)
    scorer = make_scorer(recall_score)
    estimator = create_estimator_with(predictor)
    clf = GridSearchCV(estimator, grid_parameter,
                       cv=5, scoring=scorer, n_jobs=-1,)

    # fit Logistic Regression
    classifier_name = str(estimator.named_steps["predictor"]).split('(')[0]
    title = f"\n\nTraining a GridSearchCV model with {classifier_name} predictor.."
    print(title)
    print(len(title) * "=")
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = round((time.time() - start) / 60, 2)
    print(f"Fitting took {train_time} mins.")

    # Calculate metrics
    auc_score, f_score, accuracy, precision, recall = calculate_metrics(
        clf, X_test, y_test)
    auc_score_tr, f_score_tr, accuracy_tr, precision_tr, recall_tr = calculate_metrics(
        clf, X_train, y_train)

    print("===========================")
    print(f"{classifier_name[:21]}".center(27, " "))
    print("===========================")
    print("== Test Set vs Train Set ==")
    print(f"F1 Score  : {f_score:.4f} | {f_score_tr:.4f}")
    print(f"AUC Score : {auc_score:.4f} | {auc_score_tr:.4f}")
    print(f"Accuracy  : {accuracy:.4f} | {accuracy_tr:.4f}")
    print(f"Precision : {precision:.4f} | {precision_tr:.4f}")
    print(f"Recall    : {recall:.4f} | {recall_tr:.4f}")
    print("===========================\n")

    return [
        classifier_name,
        clf,
        train_time,
        f_score,
        auc_score,
        accuracy,
        precision,
        recall,
    ]


def train_multiple_gridsearch(X_train, X_test, y_train, y_test):
    """Train multiple gridsearch cv models by creating estimators with all possible predictors. 
    Store the precitor's name, trained model, training time, and its performance in a pd.DataFrame, 
    saves, prints, and returns it

    Args:
        X_train (pd.DataFrame): train set
        X_test (pd.DataFrame): test set
        y_train (pd.Series): train labels
        y_test (pd.Series): test labels

    Returns:
        list: _description_
    """
    results_df = pd.DataFrame(
        columns=[
            "classifier_class",
            "trained_model",
            "train_time",
            "F1_Score",
            "AUC",
            "Accuracy",
            "Precision",
            "Recall",
        ]
    )
    grids_mapping = get_grids_mapping()
    for predictor, grid_parameter in grids_mapping.items():
        [
            classifier_class,
            trained_model,
            train_time,
            f_score,
            auc_score,
            accuracy,
            precision,
            recall,
        ] = train_single_gridsearch(predictor, grid_parameter, X_train, X_test, y_train, y_test)

        results_df.loc[len(results_df)] = [
            classifier_class,
            trained_model,
            train_time,
            f_score,
            auc_score,
            accuracy,
            precision,
            recall,
        ]

    results_df = results_df.sort_values(by="Recall", ascending=False)

    best_auc = results_df.AUC.values[0]
    best_recall = results_df.Recall.values[0]
    output_path = f"../trained_models/results_df_{best_auc:.2f}_AUC_{best_recall:.2f}_Rec.pkl"
    print(f"Pickling trained models dataframe at {output_path}..\n")
    results_df.to_pickle(output_path)

    # print the results to the screen
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 20)
    print(results_df.drop(columns=["trained_model"]))

    return results_df


def get_grids_mapping():
    '''Grid Parameters to consider for different predictors'''
    grids_mapping = {
        "logistic": {"predictor__C": [1.0, 0.3, 0.01, 0.003, 0.001]},
        "forest": {
            "predictor__max_depth": [15, 10, None],
            "predictor__n_estimators": [31, 51, 56, 61, 66, 71, 101],
            "predictor__max_features": ["sqrt", "log2"],
            "predictor__min_samples_split": [8, 16, 24, 32, 128],
            "predictor__criterion": ["gini", "entropy", "log_loss"],
        },
        "xgb": {
            "predictor__max_features": ["sqrt", "log2"],
            "predictor__min_samples_split": [8, 16, 24, 32, 128],
        },
        "gaussian": {"predictor__var_smoothing": [1e-9, 1e-8]},
        "quad": {"predictor__reg_param": [0.0, 0.1]},
        "decision_tree": {
            "predictor__max_depth": [25, 15, None],
            "predictor__max_features": ["sqrt", "log2"],
            "predictor__min_samples_split": [8, 16, 32, 128],
            "predictor__criterion": ["gini", "entropy", "log_loss"],
        },
    }
    return grids_mapping
