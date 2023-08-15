import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.inspection import permutation_importance

from utils_model import *


def permutation_importance_top10(model, X_test, y_test):
    start = time.time()
    result = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=0, n_jobs=-1)
    imp_mean = pd.Series(result.importances_mean, index=X_test.columns)
    imp = pd.DataFrame(result.importances, index=X_test.columns)

    perm_importance_labels = list(
        imp_mean.sort_values(ascending=False).index[:10])
    perm_importance_labels.reverse()

    perm_importance_values = imp.loc[perm_importance_labels]

    end = time.time()
    elapsed_time = (end-start)/60
    print(
        f"Elapsed time to compute permutation importances: {elapsed_time:.2f} mins")
    return [perm_importance_values, perm_importance_labels]


def feature_importance_top10(model, X_test, y_test):
    start = time.time()
    predictor = model.named_steps["predictor"]
    feature_importance = pd.Series(
        predictor.feature_importances_, index=X_test.columns)
    top10_feature_importance = feature_importance.sort_values(ascending=False)[
        :10]
    feature_importance_values = top10_feature_importance.values
    feature_importance_labels = top10_feature_importance.index

    if isinstance(predictor.estimators_[0], np.ndarray):
        std = np.std(
            [tree[0].feature_importances_ for tree in predictor.estimators_], axis=0
        )
    else:
        std = np.std(
            [tree.feature_importances_ for tree in predictor.estimators_], axis=0
        )
    feature_importance_std = pd.Series(std, index=X_test.columns)
    top10_feature_importance_std = feature_importance_std.loc[
        feature_importance_labels
    ].values

    end = time.time()
    elapsed_time = (end - start) / 60
    print(
        f"Elapsed time to compute feature importances: {elapsed_time:.2f} mins")
    return [
        feature_importance_values,
        top10_feature_importance_std,
        feature_importance_labels,
    ]


def shorten_labels(importance_labels):
    labels = []
    for label in importance_labels:
        if "_" in label:
            first_half, second_half = label.split("_")
            if len(first_half) > 25:
                fh = "".join(
                    [char for char in first_half if char.isupper()])
                label = "_".join([fh, second_half])
        labels.append(label)

    return labels


def plot_perm_and_feature_importances(model, X_test, y_test, model_type):
    feature_importance_values, top10_feature_importance_std, feature_importance_labels = feature_importance_top10(
        model, X_test, y_test)

    print("Calculating permutation importance. This could take a while..")
    perm_importance_values, perm_importance_labels = permutation_importance_top10(
        model, X_test, y_test)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x=feature_importance_values, y=shorten_labels(feature_importance_labels),
                ax=axes[0], yerr=top10_feature_importance_std)
    axes[0].set_title("Top 10 Feature Importance")
    axes[0].set_ylabel("Features")
    axes[0].set_xlim([0, 0.25])

    axes[1].boxplot(perm_importance_values.T, vert=False,
                    labels=shorten_labels(perm_importance_labels))
    axes[1].set_title("Top 10 Permutation Importance")
    axes[1].set_xlim([0, 0.025])

    fig.tight_layout()
    fig_path = f"../data/{model_type}_permutation_feature_importance.png"
    plt.savefig(fig_path)
    print(f"Figure is saved at {fig_path}")
    return fig


def calculate_stats_logit_metrics(logit, X_test, y_test):
    yhat = logit.predict(sm.add_constant(X_test))
    y_pred = list(map(round, yhat))

    auc_score = roc_auc_score(y_test, yhat)
    f_score = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return auc_score, f_score, accuracy, precision, recall


def print_logit_results(logit, X_train, X_test, y_train, y_test, imp_features):
    auc_score, f_score, accuracy, precision, recall = calculate_stats_logit_metrics(
        logit, X_test, y_test
    )
    (
        auc_score_tr,
        f_score_tr,
        accuracy_tr,
        precision_tr,
        recall_tr,
    ) = calculate_stats_logit_metrics(logit, X_train, y_train)

    labels = ["const"]
    for label in imp_features:
        labels.append(label)

    print(logit.summary(xname=labels))

    print("\n\n===========================")
    print("== Test Set vs Train Set ==")
    print(f"AUC Score : {auc_score:.4f} | {auc_score_tr:.4f}")
    print(f"Recall    : {recall:.4f} | {recall_tr:.4f}")
    print(f"F1 Score  : {f_score:.4f} | {f_score_tr:.4f}")
    print(f"Accuracy  : {accuracy:.4f} | {accuracy_tr:.4f}")
    print(f"Precision : {precision:.4f} | {precision_tr:.4f}")
    print("===========================\n")
