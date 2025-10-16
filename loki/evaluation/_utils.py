import pandas as pd
import numpy as np


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss
)


def _split_Xy(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def _make_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])
    return pre


def _default_models(class_weight='balanced', random_state='123'):

    return {
        "LogReg": LogisticRegression(max_iter=200, class_weight=class_weight),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, class_weight=class_weight, random_state=random_state
        ),
    }




def _prepare_data(datasets, target, real_data_tag = "Real",
                 n_splits=5, n_repeats=10, random_state=123):
    
    if type(datasets) != dict:
        print("data must be a dictionary")
        return


    real_data = datasets[real_data_tag]
    X_real, y_real = _split_Xy(real_data, target)

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    output = {}

    for data_name, data in datasets.items():
        output[data_name] = {"X_train":[],
                             "y_train":[],
                             "X_test":[],
                             "y_test":[]}
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_real, y_real), start=1):
            X_train, X_test = X_real.iloc[train_idx], X_real.iloc[test_idx]
            y_train, y_test = y_real.iloc[train_idx], y_real.iloc[test_idx]

            if data_name != real_data_tag:
                X_train, y_train = _split_Xy(data, target)

            output[data_name]["X_train"].append(X_train)
            output[data_name]["y_train"].append(y_train)
            output[data_name]["X_test"].append(X_test)
            output[data_name]["y_test"].append(y_test)
    
    return output


def _compute_metrics(y_true, y_pred, y_proba=None):
    out = {}
    average = "weighted"
    out["Accuracy"]  = accuracy_score(y_true, y_pred)
    out["F1"]        = f1_score(y_true, y_pred, average=average)
    out["Precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    out["Recall"]    = recall_score(y_true, y_pred, average=average)

    # # Probabilistic metrics
    # roc, brier = np.nan, np.nan
    # if y_proba is not None:
    #     try:
    #         if y_proba.ndim == 1 or y_proba.shape[1] == 1:
    #             # binary, prob of positive class
    #             roc = roc_auc_score(y_true, y_proba.ravel())
    #             brier = brier_score_loss(y_true, y_proba.ravel())
    #         else:
    #             # multiclass
    #             roc = roc_auc_score(y_true, y_proba, multi_class="ovr")
    #             # Brier: mean of per-class probabilities assigned to true class
    #             true_idx = pd.Series(y_true).map({c:i for i,c in enumerate(np.unique(y_true))}).values
    #             p_true = y_proba[np.arange(len(y_true)), true_idx]
    #             brier = brier_score_loss(y_true, p_true)  # works with integer labels
    #     except Exception:
    #         pass
    # out["ROC_AUC"] = roc
    # out["Brier"]   = brier
    return out




def _fmt(vals):
    vals = np.array(vals, dtype=float)
    return f"{np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}"













