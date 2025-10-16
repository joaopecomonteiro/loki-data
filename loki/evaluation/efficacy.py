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

            


    # for data_name, data in datasets.items():
    #     # if data_name != real_data_tag:
    #     for fold, (train_idx, test_idx) in enumerate(cv.split(X_real, y_real), start=1):
    #         X_train, X_test = X_real.iloc[train_idx], X_real.iloc[test_idx]
    #         y_train, y_test = y_real.iloc[train_idx], y_real.iloc[test_idx]

    #         for model_name, model in models.items():
    #             # --- TRTR ---
    #             trtr = pipe.fit(X_train, y_train)

    #             y_pred = trtr.predict(X_test)



    # # --- Real Data ---
    
    # real_data_output = {"Prediction": [],
    #                     "Truth": []}

    # X_test_list = []
    # y_test_list = []

    # for rep in range(n_repeats):
    #     X_train, X_test, y_train, y_test = train_test_split(
    #             X_real, y_real, test_size=test_size, stratify=y_real, random_state=random_state
    #             )
        
    #     X_test_list.append(X_test)
    #     y_test_list.append(y_test)

    #     for name, model in models.items():
    #         pipe = Pipeline([("pre", pre), ("clf", model)])

    #         trtr = pipe.fit(X_train, y_train)
    #         y_pred = trtr.predict(X_test)

    #         real_data_output["Prediction"].append(y_pred)
    #         real_data_output["Truth"].append(y_test)

    

    # synthetic_data_output = ""
    # for synthetic_data in synthetic_datas:
    #     for rep in range(n_repeats):
            
    #         for name, model in models.items():
    #             pipe = Pipeline([("pre", pre), ("clf", model)])

    #             X_synthetic, y_synthetic = _split_Xy(synthetic_data, target)
    #             tstr = pipe.fit(X_synthetic, y_synthetic)


    #             y_pred = tstr.predict(X_test)





# def machine_learning_efficacy(real_data, synthetic_data, target,
#                               preprocess=False, models=None, metrics=None,
#                               test_size=0.2, 
#                               class_weight='balanced', n_repeats=5, random_state=123):
    

#     X_real, y_real = _split_Xy(real_data, target)
#     X_synthetic, y_synthetic = _split_Xy(synthetic_data, target)



#     pre = _make_preprocessor(X=X_real) if preprocess else "passthrough"

    
#     if models is None:
#         models = _default_models(class_weight=class_weight,
#                                  random_state=random_state)
      

#     if metrics is None:
#         metrics = ["Accuracy", "F1", "Precision", "Recall", "ROC_AUC", "Brier"]
    
#     results = {
#         name: {"TRTR": {k: [] for k in metrics}, "TSTR": {k: [] for k in metrics}}
#         for name in models
#     }

#     for rep in range(n_repeats):
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_real, y_real, test_size=test_size, stratify=y_real, random_state=random_state
#         )


#         for name, model in models.items():

#             pipe = Pipeline([("pre", pre), ("clf", model)])


#             # --- TRTR ---
#             trtr = pipe.fit(X_train, y_train)

#             y_pred = trtr.predict(X_test)

#             y_proba = trtr.predict_proba(X_test) if hasattr(trtr, "predict_proba") else None

#             for k, v in _compute_metrics(y_test, y_pred, y_proba).items():
#                 results[name]["TRTR"][k].append(v)


#             # --- TSTR ---
#             tstr = pipe.fit(X_synthetic, y_synthetic)
#             y_pred = tstr.predict(X_test)
#             y_proba = tstr.predict_proba(X_test) if hasattr(tstr, "predict_proba") else None
#             for k, v in _compute_metrics(y_test, y_pred, y_proba).items():
#                 results[name]["TSTR"][k].append(v)

#     rows, data = [], []
#     for name, result in results.items():
#         for setting in ["TRTR", "TSTR"]:
#             rows.append((name, setting))
#             data.append([_fmt(result[setting][m]) for m in metrics])

#     out = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(rows, names=["Model", "Setting"]), columns=metrics)
#     return out

