import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



from ._utils import _split_Xy, _make_preprocessor, _default_models, _compute_metrics, _fmt, _prepare_data




def train_models(datasets, target, real_data_tag = "Real",
                 preprocess=False, models=None, class_weight='balanced', 
                 n_splits=5, n_repeats=10, random_state=123):
    
    data_splits = _prepare_data(datasets=datasets, target=target, 
                                real_data_tag=real_data_tag,
                                n_splits=n_splits, n_repeats=n_repeats, 
                                random_state=random_state
                                )
    


    if models is None:
        models = _default_models(class_weight=class_weight,
                                 random_state=random_state)

    output = {}
    for data_name, splits in data_splits.items():
        output[data_name] = {}

        for model_name, model in models.items():
            output[data_name][model_name] = {"prediction": [],
                                "truth": []}

            for n in range(len(splits)):
                X_train = splits["X_train"][n]
                y_train = splits["y_train"][n]
                X_test = splits["X_test"][n]
                y_test = splits["y_test"][n]

                pre = _make_preprocessor(X=X_train) if preprocess else "passthrough"

                pipe = Pipeline([("pre", pre), ("clf", model)])

                trained_pipe = pipe.fit(X_train, y_train)
                y_pred = trained_pipe.predict(X_test)

                output[data_name][model_name]["prediction"].append(y_pred)
                output[data_name][model_name]["truth"].append(y_test)

    return output


def evaluate_models(results):
    output = {}

    rows, data = [], []
    for data_name, models in results.items():
        output[data_name] = {}
        for model_name, result in models.items():
            output[data_name][model_name] = {
                "Accuracy": [],
                "F1": [],
                "Precision": [],
                "Recall": []
            }

            for n in range(len(result["prediction"])):
                pred = result["prediction"][n]
                true = result["truth"][n]

                metrics = _compute_metrics(true, pred)

                output[data_name][model_name]["Accuracy"].append(metrics["Accuracy"])
                output[data_name][model_name]["F1"].append(metrics["F1"])
                output[data_name][model_name]["Precision"].append(metrics["Precision"])
                output[data_name][model_name]["Recall"].append(metrics["Recall"])

    metrics = ["Accuracy", "F1", "Precision", "Recall"]

    rows, data = [], []
    for data_name, models in output.items():
        for model_name in models.keys():
            rows.append((data_name, model_name))
            data.append([_fmt(models[model_name][m]) for m in metrics])



    return pd.DataFrame(data, index=pd.MultiIndex.from_tuples(rows, names=["Method", "Model"]), columns=metrics)

         