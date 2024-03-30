import os
import random
from functools import partial
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score

import seaborn as sns
import matplotlib.pyplot as plt

from _2_scaling_and_dim_reduction import encode_labels, tune_base_classifiers

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

RANDOM_STATE = 2
K_FOLDS = 5

DATA_INPUT_PATH = "data/input"
DATA_OUTPUT_PATH = "data/output"

random.seed(RANDOM_STATE)


def load_500_feature_data(which: str) -> tuple[dict[str, pd.DataFrame], np.ndarray, LabelEncoder]:
    """Valid options are: `"train"` and `"test"`."""
    print(f"Loading {which} data...")
    if which not in ["train", "test"]:
        raise ValueError("Read the docs, idiot")

    omic_dfs = {}
    omic_dfs["ge"] = pd.read_csv(f"data/input/{which}/ge.csv")
    omic_dfs["cn"] = pd.read_csv(f"data/input/{which}/cn.csv")
    omic_dfs["me"] = pd.read_csv(f"data/input/{which}/me.csv")
    omic_dfs["mi"] = pd.read_csv(f"data/input/{which}/mi.csv")

    for df_name in omic_dfs:
        features = pd.read_csv(f"data/output/rfe/{df_name}_chosen_features_rfe.csv")["0"]
        omic_dfs[df_name] = omic_dfs[df_name][features.to_list()]

    # Label set
    y = pd.read_csv(f"data/input/{which}/y.csv")
    y, le = encode_labels(y)
    return omic_dfs, y, le


def load_models() -> dict[str, RandomForestClassifier]:
    """Load Scikit-learn models from pickle files at the given path"""
    print("Loading models...")
    df_names = ["ge", "cn", "me", "mi"]
    return {df_name: joblib.load(os.path.join(f"data/output/models/{df_name}.pkl")) for df_name in df_names}


def hard_vote(base_classifiers: dict[str, RandomForestClassifier], omic_samples: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Majority vote based on the mode predicted label
    Tie-break is random
    """

    # Check the base classifiers and sample work together
    assert base_classifiers.keys() == omic_samples.keys(), "dictionaries should have the same keys!"

    # predict label for each dataset/base classifier
    dfs = [df_name for df_name in base_classifiers.keys()]
    prediction_dict = {}
    for df_name in dfs:
        prediction_dict[df_name] = base_classifiers[df_name].predict(omic_samples[df_name])

    # most predicted label
    results = []
    mode_df: pd.DataFrame = pd.DataFrame(prediction_dict).mode(axis=1)
    for _, row in mode_df.iterrows():
        row.dropna(inplace=True)
        results.append(int(random.choice(row.to_list())))
    return pd.Series(results)


def soft_vote(base_classifiers: dict[str, RandomForestClassifier], omic_samples: dict[str, pd.DataFrame]) -> pd.Series:
    """Sum of predicted probability of each label, choose the highest"""
    # Check the base classifiers and sample work together
    assert base_classifiers.keys() == omic_samples.keys(), "dictionaries should have the same keys!"

    # predict label for each dataset/base classifier
    dfs = [df_name for df_name in base_classifiers.keys()]
    prediction_dict = {}
    for df_name in dfs:
        prediction_dict[df_name] = base_classifiers[df_name].predict_proba(omic_samples[df_name])

    # per class sum of class membership probabilities predicted by each base estimator
    sum_array: np.ndarray = prediction_dict[dfs[0]].copy()
    for df_name in dfs[1:]:
        sum_array += prediction_dict[df_name]

    return pd.Series(np.argmax(sum_array, axis=1))


def print_score(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    precision = partial(precision_score, average="macro", zero_division=0)
    recall = partial(recall_score, average="macro", zero_division=0)
    f1 = partial(f1_score, average="macro", zero_division=0)

    print(f"Precision: {precision(y_true=y_test, y_pred=y_pred):.3f}")
    print(f"Recall   : {recall(y_true=y_test, y_pred=y_pred):.3f}")
    print(f"F1-score : {f1(y_true=y_test, y_pred=y_pred):.3f}")
    print()


def _1_tune_base_estimators() -> None:
    """Load 500-feature datasets and train/tune models on them"""
    omic_dfs, y, _ = load_500_feature_data("train")

    model_parameters = {
        'bootstrap': [True, False],
        'max_depth': [10, 30, 50, 70, None],
        'max_features': ['log2', 'sqrt'],
        'min_samples_leaf': [1, 2, 4, 6],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [50, 100, 200, 500]
    }

    print("Gridsearch...")
    _ = tune_base_classifiers(
        omic_dfs, y,
        model_parameters,
        K_FOLDS, RANDOM_STATE,
        "data/output/tuning"
    )


def _2_save_tuned_models() -> None:
    """
    Save the trained base estimators as pickle files.
    These models have been trained on just the training set.
    """

    omic_dfs, y, _ = load_500_feature_data("train")

    print("Saving models...")
    # Parameters acquired from the hyperparameter tuning
    RFC_params_dict = {
        "ge": {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False},
        "cn": {'n_estimators': 200, 'min_samples_split':  5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth':   50, 'bootstrap': False},
        "me": {'n_estimators': 200, 'min_samples_split':  5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth':   50, 'bootstrap': False},
        "mi": {'n_estimators': 500, 'min_samples_split':  2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
    }

    for df_name, RFC_params in RFC_params_dict.items():
        RFC_model = RandomForestClassifier(random_state=RANDOM_STATE, **RFC_params)
        RFC_model.fit(omic_dfs[df_name], y)

        joblib.dump(RFC_model, f"data/output/models/{df_name}.pkl")


def _3_test_performance() -> None:
    """Print the test performance for each individual model and both voting strategies"""

    omic_test, y_test, _ = load_500_feature_data("test")
    model_dict = load_models()

    omic_test.pop("cn")
    model_dict.pop("cn")

    for df_name in model_dict:
        print(f"{df_name}:")
        y_pred = model_dict[df_name].predict(omic_test[df_name])
        print_score(y_test, y_pred)

    print("Hard vote:")
    y_pred_hard = hard_vote(model_dict, omic_test)
    print_score(y_test, y_pred_hard)

    print("Soft vote:")
    y_pred_soft = soft_vote(model_dict, omic_test)
    print_score(y_test, y_pred_soft)


def _4_get_permutation_importance() -> None:
    omic_test, y_test, _ = load_500_feature_data("test")
    models = load_models()

    scorer = make_scorer(f1_score, average="macro", zero_division=0)

    for df_name, model in models.items():
        print(df_name)
        start_time = time.time()
        result = permutation_importance(
            model,
            omic_test[df_name], y_test,
            scoring=scorer, n_repeats=10,
            random_state=RANDOM_STATE, n_jobs=4
        )

        elapsed_time = time.time() - start_time
        print(f"Elapsed time to compute the importances ({df_name}): {elapsed_time:.3f} seconds")

        mean = pd.Series(result.importances_mean, index=model.feature_names_in_, name="mean")
        std = pd.Series(result.importances_std, index=model.feature_names_in_, name="std")

        forest_importances = pd.concat([mean, std], axis=1)
        forest_importances.to_csv(f"data/output/permutation_importance/{df_name}_importances.csv", index="feature")


def _5_plot_permutation_importances() -> None:
    ge = pd.read_csv("data/output/permutation_importance/ge_importances.csv")
    # cn = pd.read_csv("data/output/permutation_importance/cn_importances.csv")
    # me = pd.read_csv("data/output/permutation_importance/me_importances.csv")
    # mi = pd.read_csv("data/output/permutation_importance/mi_importances.csv")

    # Plot
    sns.set_theme(font_scale=1, font="serif", style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot([-1, 1], [0, 0], "k", zorder=1)
    plt.plot([0, 0], [-1, 1], "k", zorder=2)
    sns.kdeplot(data=ge, x="mean", y="std", cmap="viridis", zorder=3)
    plt.scatter(ge["mean"], ge["std"], zorder=4)

    # ax = sns.barplot(x='feature', y='mean', data=ge, errorbar='sd')
    # ax.tick_params(axis='x', labelrotation=45)
    # # plt.errorbar(x=ge['feature'], y=ge['mean'], yerr=ge['std'])  # Add custom error bars
    plt.xlabel('Mean Value')
    plt.ylabel('Standard Deviation')
    # plt.title('Mean Values with Standard Deviation Error Bars')
    plt.xticks(np.arange(-0.006, 0.012, 0.003))
    plt.yticks(np.arange(-0.002, 0.009, 0.001))
    plt.xlim(-0.006, 0.012)
    plt.ylim(-0.002, 0.008)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-3, -3), useMathText=True)
    plt.grid(True, "both", zorder=0)
    plt.show()


def print_most_important_features() -> None:
    ge = pd.read_csv("data/output/permutation_importance/ge_importances.csv").sort_values(by="mean", ascending=False)
    cn = pd.read_csv("data/output/permutation_importance/cn_importances.csv").sort_values(by="mean", ascending=False)
    # me = pd.read_csv("data/output/permutation_importance/me_importances.csv").sort_values(by="mean", ascending=False)
    # mi = pd.read_csv("data/output/permutation_importance/mi_importances.csv").sort_values(by="mean", ascending=False)

    ge["feature"] = ge["feature"].str.lower()
    cn["feature"] = cn["feature"].str.lower()

    tmp = pd.merge(ge[ge["feature"].isin(cn["feature"])], cn[cn["feature"].isin(ge["feature"])], on="feature")
    # print(tmp[(tmp["mean_x"] > 0) & (tmp["mean_y"] > 0)])
    print(tmp)

    # print(top_100_ge.head())
    # print(top_100_cn.head())
    # print(top_100_me.head())
    # print(top_100_mi.head())


if __name__ == "__main__":
    print_most_important_features()
