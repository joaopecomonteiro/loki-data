import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import warnings
from time import time


from ._evaluation_utils import _split_Xy, _make_preprocessor, _default_models, _compute_metrics, _fmt, _prepare_data

from loki._utils import lokiError


class Trainer:

    def __init__(self, models=None, 
                 class_weight='balanced', n_splits=5, n_repeats=10, random_state=123, 
                 preprocess_data=True, verbose=0):

        # self.datasets = datasets
        # self.target = target
        self.class_weight = class_weight
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.random_state = random_state
        self.models = models
        self.preprocess_data = preprocess_data
        self.verbose = verbose



    def train(self, datasets=None, target=None):
        self.datasets = datasets
        self.target = target

        self.data_splits = _prepare_data(datasets=self.datasets, target=self.target,
                                n_splits=self.n_splits, n_repeats=self.n_repeats, 
                                random_state=self.random_state
                                )

        self.predictions = {}

        self.trained_models = {}
        for dataset_name, splits in self.data_splits.items():
            if self.verbose >= 1:
                print(f"---- Starting the training for the {dataset_name} dataset ----\n")
            
            if self.verbose >= 2:
                start_dataset = time()
                
            self.predictions[dataset_name] = {}



            for model_name, model in self.models.items():
                if self.verbose >= 2:
                    print(f"-- Model: {model_name} --")
                    start_model = time()

                self.predictions[dataset_name][model_name] = {"prediction": [],
                                    "true": []}

                for n in range(len(splits)):
                    X_train = splits["X_train"][n]
                    y_train = splits["y_train"][n]
                    X_test = splits["X_test"][n]
                    y_test = splits["y_test"][n]

                    pre = _make_preprocessor(X=X_train) if self.preprocess_data else "passthrough"

                    pipe = Pipeline([("pre", pre), ("clf", model)])

                    trained_pipe = pipe.fit(X_train, y_train)
                    y_pred = trained_pipe.predict(X_test)

                    self.predictions[dataset_name][model_name]["prediction"].append(y_pred)
                    self.predictions[dataset_name][model_name]["true"].append(y_test)

                if self.verbose >= 2:
                    print(f"Took {time()-start_model} seconds to train {model_name} with the {dataset_name} dataset.\n")
            
            if self.verbose >= 2:
                print(f"Took {time()-start_dataset} seconds to train all the models with the {dataset_name} dataset.\n")


    def evaluate(self):

        self.results = {}
        for data_name, models in self.predictions.items():
            self.results[data_name] = {}
            for model_name, result in models.items():
                self.results[data_name][model_name] = {
                    "Accuracy": [],
                    "F1": [],
                    "Precision": [],
                    "Recall": []
                }

                for n in range(len(result["prediction"])):
                    pred = result["prediction"][n]
                    true = result["true"][n]

                    metrics = _compute_metrics(true, pred)

                    self.results[data_name][model_name]["Accuracy"].append(metrics["Accuracy"])
                    self.results[data_name][model_name]["F1"].append(metrics["F1"])
                    self.results[data_name][model_name]["Precision"].append(metrics["Precision"])
                    self.results[data_name][model_name]["Recall"].append(metrics["Recall"])
        
        self.compute_results_table()

        return self.results_table



    def compute_results_table(self):
        metrics = ["Accuracy", "F1", "Precision", "Recall"] 

        rows, data = [], []
        for data_name, models in self.results.items():
            for model_name in models.keys():
                rows.append((data_name, model_name))
                data.append([_fmt(models[model_name][m]) for m in metrics])

        self.results_table = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(rows, names=["Method", "Model"]), columns=metrics)



    @property
    def datasets(self):
        return self._datasets
    @datasets.setter
    def datasets(self, val):
        if val is None:
            raise lokiError("Trainer is missing requiered attribute 'datasets'.", '0001')

        if type(val) != dict:
            raise lokiError("datasets must be a dictionary", '0002')

        self._datasets = val

        if hasattr(self, '_target'):
            warnings.warn("target attribute should also be updated.")


    @property
    def target(self):
        return self._target
    @target.setter
    def target(self, val):
        if val is None:
            raise lokiError("Trainer is missing requiered attribute 'target'.", '0001')

        self._target = val


    @property
    def n_repeats(self):
        return self._n_repeats
    @n_repeats.setter
    def n_repeats(self, val):
        self._n_repeats = val


    @property
    def models(self):
        return self._models
    @models.setter
    def models(self, val):

        if val is None:
            val = _default_models(class_weight=self.class_weight,
                            random_state=self.random_state)
        
        if type(val) != dict:
            raise lokiError("models must be a dictionary", '0002')

        self._models = val


    @property
    def n_splits(self):
        return self._n_splits
    @n_splits.setter
    def n_splits(self, val):
        self._n_splits = val


    @property
    def random_state(self):
        return self._random_state
    @random_state.setter
    def random_state(self, val):
        self._random_state = val


    @property
    def class_weight(self):
        return self._class_weight
    @class_weight.setter
    def class_weight(self, val):
        self._class_weight = val


    @property
    def n_repeats(self):
        return self._n_repeats
    @n_repeats.setter
    def n_repeats(self, val):
        self._n_repeats = val



