import sys
from typing import List, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from optic_metrology.feature import FeatureType
from optic_metrology.meta_info import ModelMetaInfo
from optic_metrology.model import Model

from optic_metrology.reader import DataSetReader, InmemoryDataSet
from sklearn import metrics, preprocessing, linear_model


def train(training_data_path: str, target_name: str, random_state: Optional[int] = None):
    reader = DataSetReader()
    dataset = reader.read(training_data_path)
    X_train, X_test, y_train, y_test = dataset.sample(target_name, random_state=0)
    class_labels = None
    if dataset.get_feature_type(target_name) == FeatureType.CATEGORICAL:
        le = preprocessing.LabelEncoder()
        y_train = le.fit_transform(y_train)
        le = preprocessing.LabelEncoder()
        y_test = le.fit_transform(y_test)
        class_labels = le.classes_
    models = get_models(dataset, class_labels=class_labels, random_state=random_state)
    for model in models:
        model.fit(X_train, y_train)
    best_model = None
    best_metric = 0
    for model in models:
        preds = model.predict(X_test)
        metric = metrics.accuracy_score(preds, y_test)
        if metric >= best_metric:
            best_model = model
            best_metric = metric
    return best_model


def train_single(training_data_path: str, target_name: str, model_meta_info: ModelMetaInfo, random_state: Optional[int] = None):
    reader = DataSetReader()
    dataset = reader.read(training_data_path)
    model = Model(model_meta_info, dataset.metainfo, target_name)
    X_train, X_test, y_train, y_test = dataset.sample(target_name, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metric = metrics.accuracy_score(preds, y_test)
    return model, metric


def get_models(dataset: InmemoryDataSet, class_labels: Optional[np.ndarray] = None, random_state: Optional[int] = None):
    if class_labels is None:
        # regression todo
        return []
    return [
        XGBClassifier(),
        lgb.LGBMClassifier(num_class=len(class_labels), random_state=random_state),
        linear_model.SGDClassifier(random_state=random_state),
    ]


if __name__ == '__main__':
    train(sys.argv[1], sys.argv[2], random_state=123422)

