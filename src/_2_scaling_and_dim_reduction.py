import os

import pandas as pd
import numpy as np

from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV, RFE
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 2
K_FOLDS = 3


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_trans = le.fit_transform(y.values.ravel())
    return y_trans, le


def tune_base_classifiers(
    omic_train: dict[str, pd.DataFrame],
    y_train: pd.Series,
    parameters: dict[str, list],
    k_folds: int,
    random_state: int,
    path: str,
    extension: str = ""
) -> dict[str, RandomForestClassifier]:
    """Tune the base classifiers using a stratified k-fold gridsearch cross-validation"""

    cv = StratifiedKFold(n_splits=k_folds)

    models = {
        "ge": RandomForestClassifier(random_state=random_state),
        "cn": RandomForestClassifier(random_state=random_state),
        "me": RandomForestClassifier(random_state=random_state),
        "mi": RandomForestClassifier(random_state=random_state),
    }
    tuned_models = {}

    scorers = {
        "precision": make_scorer(precision_score, average="macro", zero_division=0),
        "recall": make_scorer(recall_score, average="macro", zero_division=0),
        "f1": make_scorer(f1_score, average="macro", zero_division=0)
    }

    for df_name in omic_train.keys():
        print(f"Tuning {df_name + extension}...")
        tuned_models[df_name] = RandomizedSearchCV(
            models[df_name],
            parameters,
            n_iter=100,
            cv=cv,
            n_jobs=4,
            verbose=1,
            scoring=scorers,
            refit="f1",
            random_state=random_state
        ).fit(omic_train[df_name], y_train)

        print("", flush=True)

        idx = tuned_models[df_name].best_index_
        res = tuned_models[df_name].cv_results_
        print(f"best parameters ({df_name + extension}): (index: {idx})")
        print(res['params'][idx])
        print(
            f"Precision: {float(res['mean_test_precision'][idx]):.3f} +/- {float(res['std_test_precision'][idx]):.3f}")
        print(f"Recall   : {float(res['mean_test_recall'][idx]):.3f} +/- {float(res['std_test_recall'][idx]):.3f}")
        print(f"F1-score : {float(res['mean_test_f1'][idx]):.3f} +/- {float(res['std_test_f1'][idx]):.3f}")
        pd.DataFrame(tuned_models[df_name].cv_results_).to_csv(os.path.join(path, f"{df_name + extension}.csv"))

    return tuned_models


def _1_scale() -> None:
    """Scale all data except CN"""
    ge = pd.read_csv("data/input/omics/ge.csv", index_col=0).set_index("sample", drop=True)
    ge_cols = ge.columns
    cn = pd.read_csv("data/input/omics/cn.csv", index_col=0).set_index("sample", drop=True)
    me = pd.read_csv("data/input/omics/me.csv", index_col=0).set_index("sample", drop=True)
    me_cols = me.columns
    mi = pd.read_csv("data/input/omics/mi.csv", index_col=0).set_index("sample", drop=True)
    mi_cols = mi.columns

    ge_scale = StandardScaler()
    pd.DataFrame(ge_scale.fit_transform(ge), columns=ge_cols, index=ge.index.to_series()).to_csv(
        "data/input/omics/ge.csv")
    print("ge")

    cn = cn.astype(int)
    cn.to_csv(
        "data/input/omics/cn.csv")
    print("cn")

    me_scale = StandardScaler()
    pd.DataFrame(me_scale.fit_transform(me), columns=me_cols, index=me.index.to_series()).to_csv(
        "data/input/omics/me.csv")
    print("me")

    mi_scale = StandardScaler()
    pd.DataFrame(mi_scale.fit_transform(mi), columns=mi_cols, index=mi.index.to_series()).to_csv(
        "data/input/omics/mi.csv")
    print("mi")


def _2_split_train_test() -> None:
    """Make training/testing split"""
    omic_dfs: dict[str, pd.DataFrame] = {}
    omic_dfs["ge"] = pd.read_csv("data/input/omics/ge.csv", index_col=0)
    omic_dfs["cn"] = pd.read_csv("data/input/omics/cn.csv", index_col=0)
    omic_dfs["me"] = pd.read_csv("data/input/omics/me.csv", index_col=0)
    omic_dfs["mi"] = pd.read_csv("data/input/omics/mi.csv", index_col=0)

    labels = pd.read_csv("data/input/omics/y.csv").set_index("sample", drop=True)
    labels.sort_index(inplace=True)
    y = labels["label"]

    for df_name in omic_dfs:
        omic_dfs[df_name] = omic_dfs[df_name].sort_index()

        X_train, X_test, y_train, y_test = train_test_split(
            omic_dfs[df_name], y,
            test_size=0.25,
            random_state=RANDOM_STATE,
            shuffle=True, stratify=y
        )
        X_train.to_csv(f"data/input/train/{df_name}_no_me.csv", index=False)
        X_test.to_csv(f"data/input/test/{df_name}_no_me.csv", index=False)
        y_train.to_csv("data/input/train/y_no_me.csv", index=False)
        y_test.to_csv("data/input/test/y_no_me.csv", index=False)


def _3_RFECV_tuning_and_execution(model_parameters: dict[str, list], extension: str = "") -> None:
    """First perform a hyperparameter tuning for the estimators used in the RFECV, then perform the RFECV"""
    omic_dfs: dict[str, pd.DataFrame] = {}
    omic_dfs["ge"] = pd.read_csv(f"data/input/train/ge{extension}.csv")
    omic_dfs["cn"] = pd.read_csv(f"data/input/train/cn{extension}.csv")
    if extension == "":
        omic_dfs["me"] = pd.read_csv(f"data/input/train/me.csv")
    omic_dfs["mi"] = pd.read_csv(f"data/input/train/mi{extension}.csv")

    y = pd.read_csv(f"data/input/train/y{extension}.csv")
    y, le = encode_labels(y)

    print("Gridsearch...")

    tuned_models = tune_base_classifiers(
        omic_dfs, y,
        model_parameters,
        K_FOLDS, RANDOM_STATE,
        "data/output/rfe/tuning",
        extension
    )

    # tuned_models = {
    #     "ge": {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth':   70, 'bootstrap': False},
    #     "cn": {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None, 'bootstrap': False},
    #     "me": {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth':   10, 'bootstrap': False},
    #     "mi": {'n_estimators':  50, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth':   70, 'bootstrap': False},
    # }

    print("RFECV...")
    cv = StratifiedKFold(n_splits=K_FOLDS)
    f1 = make_scorer(f1_score, average="macro", zero_division=0)
    for df_name in omic_dfs:
        print(df_name + extension)
        idx = tuned_models[df_name].best_index_
        res = tuned_models[df_name].cv_results_
        rfecv = RFECV(
            RandomForestClassifier(**res["params"][idx], random_state=RANDOM_STATE),
            step=0.1, min_features_to_select=1,
            cv=cv, scoring=f1, verbose=3
        ).fit(omic_dfs[df_name], y)

        print(pd.Series(rfecv.ranking_).value_counts())

        chosen_features = omic_dfs[df_name].columns.to_series()[rfecv.support_]
        chosen_features.to_csv(f"data/output/rfe/{df_name + extension}_chosen_features_rfecv.csv", index=False)
        pd.DataFrame(rfecv.cv_results_).to_csv(f"data/output/rfe/{df_name + extension}_rfecv.csv")
        print(f"done with {df_name + extension}")


def _4_RFE_dim_reduction(extension: str = "") -> None:
    """RFECV will select the features it thinks are optimal. But we don't want that. We just want 500 features."""
    omic_dfs: dict[str, pd.DataFrame] = {}
    omic_dfs["ge"] = pd.read_csv(f"data/input/train/ge{extension}.csv")
    omic_dfs["cn"] = pd.read_csv(f"data/input/train/cn{extension}.csv")
    if extension == "":
        omic_dfs["me"] = pd.read_csv(f"data/input/train/me.csv")
    omic_dfs["mi"] = pd.read_csv(f"data/input/train/mi{extension}.csv")

    y = pd.read_csv(f"data/input/train/y{extension}.csv")
    y, le = encode_labels(y)

    # parameters are the same as the ones obtained from the RFECV tuning
    tuned_models = {
        "ge": {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth':   70, 'bootstrap': False},
        "cn": {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None, 'bootstrap': False},
        "me": {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth':   10, 'bootstrap': False},
        "mi": {'n_estimators':  50, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth':   70, 'bootstrap': False},
    }

    print("RFE...")
    for df_name in omic_dfs:
        print(df_name + extension)
        rfecv = RFE(
            RandomForestClassifier(**tuned_models[df_name], random_state=RANDOM_STATE),
            step=0.1,
            n_features_to_select=500,
            verbose=3
        ).fit(omic_dfs[df_name], y)

        chosen_features = omic_dfs[df_name].columns.to_series()[rfecv.support_]
        chosen_features.to_csv(
            f"data/output/rfe/{df_name + extension}_chosen_features_rfe.csv",
            index=False
        )
        print(f"done with {df_name + extension}")


def _5_plot_rfecv_results() -> None:
    ge = pd.read_csv("data/output/rfe/ge_rfecv.csv")
    ge = ge[["split0_test_score", "split1_test_score", "split2_test_score"]].T
    cn = pd.read_csv("data/output/rfe/cn_rfecv.csv")
    cn = cn[["split0_test_score", "split1_test_score", "split2_test_score"]].T
    me = pd.read_csv("data/output/rfe/me_rfecv.csv")
    me = me[["split0_test_score", "split1_test_score", "split2_test_score"]].T
    mi = pd.read_csv("data/output/rfe/mi_rfecv.csv")
    mi = mi[["split0_test_score", "split1_test_score", "split2_test_score"]].T

    ge['Method'] = 'GE'
    cn['Method'] = 'CN'
    me['Method'] = 'ME'
    mi['Method'] = 'MI'

    # Concatenate the DataFrames
    data = pd.concat([ge, cn, me, mi])

    # Melt the DataFrame to have 'Method' as a column
    data_melted = data.melt(id_vars=['Method'], var_name='Split', value_name='Test Score')

    # Plot
    sns.set_theme(font_scale=2, font="serif", style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_melted, x='Method', y='Test Score', hue='Split')

    plt.yticks([i/10 for i in range(11)])
    plt.ylabel('F$_1$-score')
    plt.legend(
        title='Split',
        bbox_to_anchor=(0.12, -0.35, 0.75, .102),
        loc='center',
        ncol=6,
        mode="expand",
        borderaxespad=0.
    )
    plt.grid(True, "both", "y")
    plt.show()


if __name__ == "__main__":
    # Parameter options for tuning
    # model_parameters = {
    #     'bootstrap': [True, False],
    #     'max_depth': [10, 30, 70, None],
    #     'max_features': ['log2', 'sqrt'],
    #     'min_samples_leaf': [1, 2, 4],
    #     'min_samples_split': [2, 5, 10],
    #     'n_estimators': [50, 100, 200, 500, 1000]
    # }

    _5_plot_rfecv_results()
