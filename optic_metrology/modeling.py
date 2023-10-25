import json
import logging
import sys
from typing import Iterable, List, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
from optic_metrology.feature import FeatureType, FeaturesMetainfo
from optic_metrology.meta_info import ModelMetaInfo, ModelType, VertexMetaInfo
from optic_metrology.model import Model

from optic_metrology.reader import DataSetReader, InmemoryDataSet
from sklearn import metrics

CLASSIFICATION_ESTIMATORS = [
    {
        'clazz': 'sklearn.linear_model.SGDClassifier',
        'hyper_parameters': {},
        'method': 'predict',
    },
    {
        'clazz': 'xgboost.XGBClassifier',
        'hyper_parameters': {},
        'method': 'predict',
    },
    {
        'clazz': 'sklearn.svm.SVC',
        'hyper_parameters': {
            'gamma': 2,
            'C': 1,
        },
        'method': 'predict',
    },
    {
        'clazz': 'lightgbm.LGBMClassifier',
        'hyper_parameters': {
            'verbose': -1,
        },
        'method': 'predict',
    },
    {
        'clazz':  'sklearn.tree.DecisionTreeClassifier',
        'hyper_parameters': {
            'max_depth': 5,
        },
        'method': 'predict',
    },
    {
        'clazz': 'sklearn.ensemble.RandomForestClassifier',
        'hyper_parameters': {
            'max_depth':5,
            'n_estimators': 10, 
            'max_features':1,
        },
        'method': 'predict',
    },
    {
        'clazz': 'sklearn.ensemble.AdaBoostClassifier',
        'hyper_parameters': {},
        'method': 'predict',
    },
]

NUMERIC_IMPUTATIONS  = [
    {
        'clazz': 'sklearn.impute.SimpleImputer',
        'hyper_parameters': {},
        'method': 'transform',
    },
    {
        'clazz': 'sklearn.impute.KNNImputer',
        'hyper_parameters': {
            'n_neighbours': 2,
        },
        'method': 'transform',
    }
]

CATEGORICAL_ENCODERS = [
    {
        'clazz': 'sklearn.preprocessing.OrdinalEncoder',
        'hyper_parameters': {
            'handle_unknown': 'use_encoded_value',
            'unknown_value': -9999,
        },
        'method': 'transform',
    },
    {
        'clazz': 'sklearn.preprocessing.OneHotEncoder',
        'hyper_parameters': {
            'handle_unknown': 'ignore',
        },
        'method': 'transform',
    },
]

LOGGER = logging.getLogger(__name__)


class ModelGenerator(object):

    def generate(self, features_meta_info: FeaturesMetainfo, model_type: ModelType) -> List[ModelMetaInfo]:
        return self.permutations(features_meta_info, model_type)
    
    def permutations(self, features_meta_info: FeaturesMetainfo, model_type: ModelType) -> List[ModelMetaInfo]:
        preprocessing_steps = []
        estimators = None
        if model_type in [ModelType.BINARY, ModelType.MULTICLASS]:
            estimators = CLASSIFICATION_ESTIMATORS
        types_count = features_meta_info.types_count
        types = []
        if FeatureType.NUMERIC in types_count:
            preprocessing_steps.append(NUMERIC_IMPUTATIONS)
            types.append(FeatureType.NUMERIC)
        if FeatureType.CATEGORICAL in types_count:
            preprocessing_steps.append(CATEGORICAL_ENCODERS)
            types.append(FeatureType.CATEGORICAL)
        preprocessing_combs = []
        self.preprocessing_combs_recursive(preprocessing_steps, 0, [], preprocessing_combs)
        result = []
        for est_dict in estimators:
            parents = [str(i) for i in range(len(preprocessing_steps))]
            est_dict = dict(est_dict)
            est_dict['name'] = 'Vertex_{}'.format(len(preprocessing_steps))
            est_dict['uid'] = '{}'.format(len(preprocessing_steps))
            for preprocessing_comb in preprocessing_combs:
                est_vertex = VertexMetaInfo.from_dict(est_dict)
                model = ModelMetaInfo()
                for i in range(len(preprocessing_comb)):
                    sources = [types[i]]
                    task_index = preprocessing_comb[i]
                    prep_dict = dict(preprocessing_steps[i][task_index])
                    prep_dict['name'] = 'Vertex_{}'.format(i)
                    prep_dict['uid'] = str(i)
                    model.add(VertexMetaInfo.from_dict(prep_dict), feature_selection=sources)
                model.add(est_vertex, parents_uids=parents)
                result.append(model)
        return result

    def preprocessing_combs_recursive(self, steps: List[List[dict]], current: int, path: List[int], results: List[List[int]]):
        if current == len(steps):
            results.append(list(path))
            return
        for i in range(len(steps[current])):
            path.append(i)
            self.preprocessing_combs_recursive(steps, current + 1, path, results)
            del path[-1]


def train(dataset: InmemoryDataSet, model_type: Optional[ModelType] = None, target_name: Optional[str] = None, random_state: Optional[int] = None):
    model_generator = ModelGenerator()
    features_info = dataset.metainfo.subset([target_name], include=False)
    if not model_type and target_name:
        model_type = ModelType.REGRESSION if dataset.get_feature_type(target_name) == FeatureType.NUMERIC else ModelType.MULTICLASS
    elif not model_type:
        model_type = ModelType.ANOMALY_DETECTION
    models_infos = model_generator.generate(features_info, model_type)
    LOGGER.info("Models to train: [%s]", len(models_infos))
    results = []
    for model_meta_info in models_infos:
        model, metric = train_single(dataset, model_meta_info, model_type, random_state=random_state, target_name=target_name)
        results.append((model, metric))
        LOGGER.info("Progress: %s/%s",len(results), len(models_infos))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results


def train_single(dataset: InmemoryDataSet, model_meta_info: ModelMetaInfo, model_type: ModelType, target_name: Optional[str] = None, random_state: Optional[int] = None):
    model = Model(model_meta_info, dataset.metainfo, model_type, target_name)
    X_train, X_test, y_train, y_test = dataset.sample(target_name, random_state=0)
    model.fit(X_train, y_train, random_state=random_state)
    preds = model.predict(X_test)
    metric = metrics.accuracy_score(preds, y_test)
    return model, metric

def train_predefined(dataset: InmemoryDataSet, target_name: str, models_list: str, random_state: Optional[int] = None):
    models_infos = []
    for model_dict in models_list:
        models_infos.append(ModelMetaInfo.from_dict(model_dict))
    LOGGER.info("Models to train: [%s]", len(models_infos))
    results = []
    for model_meta_info in models_infos:
        model, metric = train_single(dataset, target_name, model_meta_info, random_state)
        results.append((model, metric))
        LOGGER.info("Progress: %s/%s",len(results), len(models_infos))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results