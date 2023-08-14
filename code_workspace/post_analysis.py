from sklearn.inspection import permutation_importance
from utils_model import * 
import matplotlib.pyplot as plt
import seaborn as sns 

def permutation_importance_top10(model, X_test, y_test):
    start = time.time()
    result = permutation_importance(model, X_test, y_test, n_repeats=2, random_state=0, n_jobs=-1)
    imp_mean = pd.Series(result.importances_mean, index=X_test.columns)
    imp = pd.DataFrame(result.importances, index=X_test.columns)

    perm_importance_labels = list(imp_mean.sort_values(ascending=False).index[:10])
    perm_importance_labels.reverse()

    perm_importance_values = imp.loc[perm_importance_labels]

    end = time.time()
    elapsed_time = (end-start)/60
    print(f"Elapsed time to compute permutation importances: {elapsed_time:.2f} mins")
    return [perm_importance_values, perm_importance_labels]


def feature_importance_top10(model, X_test, y_test):
    start = time.time()
    predictor = model.named_steps["predictor"]
    feature_importance = pd.Series(predictor.feature_importances_, index=X_test.columns)
    top10_feature_importance = feature_importance.sort_values(ascending=False)[:10]
    feature_importance_values = top10_feature_importance.values
    feature_importance_labels = top10_feature_importance.index
    
    std = np.std([tree.feature_importances_ for tree in predictor.estimators_], axis=0)
    feature_importance_std = pd.Series(std, index=X_test.columns)
    top10_feature_importance_std = feature_importance_std.loc[feature_importance_labels]

    end = time.time()
    elapsed_time = (end-start)/60
    print(f"Elapsed time to compute feature importances: {elapsed_time:.2f} mins")
    return [feature_importance_values, top10_feature_importance_std, feature_importance_labels]


def plot_perm_and_feature_importances(model, X_test, y_test):
    print("Calculating permutation importance. This could take a while..")
    perm_importance_values, perm_importance_labels = permutation_importance_top10(model, X_test, y_test)
    feature_importance_values, top10_feature_importance_std, feature_importance_labels = feature_importance_top10(model, X_test, y_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x=feature_importance_values, y=feature_importance_labels, ax=axes[0], yerr = top10_feature_importance_std)
    axes[0].set_title("Top 10 Feature Importance")
    axes[0].set_ylabel("Features")

    axes[1].boxplot(perm_importance_values.T, vert=False, labels=perm_importance_labels)
    axes[1].set_title("Top 10 Permutation Importance")

    fig.tight_layout()
    fig_path = "../data/permutation_feature_importance.png"
    plt.savefig(fig_path)
    print(f"Figure is saved at {fig_path}")
    return fig 
