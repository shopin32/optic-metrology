import numpy as np
import pytest
from sklearn import linear_model
from optic_metrology.feature_impact import compute_feature_impact
from optic_metrology.meta_info import ModelMetaInfo, ModelType
from optic_metrology.reader import DataSetReader
from tests.utils import get_test_file
from optic_metrology.modeling import train, train_single
from sklearn.cluster import KMeans


def test_train_with_model_generation():
    # arrange
    dataset_path = get_test_file('iris_fisher.xlsx')
    target_name = 'Вид ірису'
    reader = DataSetReader()
    dataset = reader.read(dataset_path)

    # Act
    models = train(dataset, target_name=target_name, random_state=123422)

    # Assert
    assert models is not None
    assert len(models) == 14
    assert abs(models[0][1] - 0.933333) < 0.00001 


def test_readmitted():
    # arrange
    dataset_path = get_test_file('diabetes.csv')
    target_name = 'readmitted'
    reader = DataSetReader()
    dataset = reader.read(dataset_path)

    # Act
    models = train(dataset, target_name=target_name, random_state=123422)

    # Assert
    assert models is not None
    assert len(models) == 28
    assert abs(models[0][1] - 0.56462) < 0.00001


def test_train_single_model():
    # arrange
    dataset_path = get_test_file('iris_fisher.xlsx')
    reader = DataSetReader()
    target_name = 'Вид ірису'
    model = ModelMetaInfo.from_dict({
        'vertices': [
            {
                'uid': '0',
                'name': 'NumericMissingValuesHandler',
                'clazz': 'sklearn.impute.SimpleImputer',
                'method': 'transform',
                'parents': [],
                'hyper_parameters': {
                    'missing_values': np.nan,
                    'strategy': 'mean',
                }
            },
            {
                'uid': '1',
                'parents': ['0'],
                'name': 'etimator',
                'clazz': 'xgboost.XGBClassifier',
                'method': 'predict',
                'hyper_parameters': {
                    'use_label_encoder': False, 
                    'eval_metric': 'mlogloss',
                }
            },
        ],
        'sources': {
            '0': ['NUMERIC']
        }
    })

    # Act
    dataset = reader.read(dataset_path)
    model, _ = train_single(dataset, model, target_name=target_name, random_state=123)
    result = model.predict(dataset.get_df(), show_probs=True)
    feature_impact = compute_feature_impact(dataset, model)

    # Assert
    assert result.shape[1] == 3

@pytest.mark.parametrize(
    'train_dataset,model_dict',
    [
        (
            'iris_fisher.xlsx',
            {
                'vertices': [
                    {
                        'uid': '0',
                        'name': 'NumericMissingValuesHandler',
                        'clazz': 'sklearn.impute.SimpleImputer',
                        'method': 'transform',
                        'parents': [],
                        'hyper_parameters': {
                            'missing_values': np.nan,
                            'strategy': 'mean',
                        }
                    },
                    {
                        'uid': '1',
                        'parents': ['0'],
                        'name': 'etimator',
                        'clazz': 'sklearn.cluster.KMeans',
                        'method': 'predict',
                        'hyper_parameters': {
                            'n_clusters': 3, 
                        }
                    },
                ],
                'sources': {
                    '0': ['NUMERIC']
                }
            }
        ),
        (
            'iris_fisher.xlsx',
            {
                'vertices': [
                    {
                        'uid': '0',
                        'name': 'NumericMissingValuesHandler',
                        'clazz': 'sklearn.impute.SimpleImputer',
                        'method': 'transform',
                        'parents': [],
                        'hyper_parameters': {
                            'missing_values': np.nan,
                            'strategy': 'mean',
                        }
                    },
                    {
                        'uid': '1',
                        'parents': ['0'],
                        'name': 'etimator',
                        'clazz': 'sklearn.cluster.OPTICS',
                        'method': 'fit_predict',
                        'hyper_parameters': {}
                    },
                ],
                'sources': {
                    '0': ['NUMERIC']
                }
            }
        ),
        (
            'iris_fisher.xlsx',
            {
                'vertices': [
                    {
                        'uid': '0',
                        'name': 'NumericMissingValuesHandler',
                        'clazz': 'sklearn.impute.SimpleImputer',
                        'method': 'transform',
                        'parents': [],
                        'hyper_parameters': {
                            'missing_values': np.nan,
                            'strategy': 'mean',
                        }
                    },
                    {
                        'uid': '1',
                        'parents': ['0'],
                        'name': 'etimator',
                        'clazz': 'sklearn.cluster.DBSCAN',
                        'method': 'fit_predict',
                        'hyper_parameters': {
                            'eps': 0.1,
                        }
                    },
                ],
                'sources': {
                    '0': ['NUMERIC']
                }
            }
        ),
    ]
)
def test_train_unsupervised_clustering(train_dataset: str, model_dict: dict):
    # arrange
    dataset_path = get_test_file(train_dataset)
    reader = DataSetReader()
    model = ModelMetaInfo.from_dict(model_dict)

    # Act
    dataset = reader.read(dataset_path)
    model, _ = train_single(dataset, model, model_type=ModelType.UNSUPERVISED_CLUSTERING, random_state=123)
    result = model.predict(dataset.get_df())

    # Assert
    assert result.shape[1] == 1
    assert result[result.columns[0]].dtype in [np.int32, np.int64]

@pytest.mark.parametrize(
    'train_dataset,model_dict',
    [
        (
            'iris_fisher.xlsx',
            {
                'vertices': [
                    {
                        'uid': '0',
                        'name': 'NumericMissingValuesHandler',
                        'clazz': 'sklearn.impute.SimpleImputer',
                        'method': 'transform',
                        'parents': [],
                        'hyper_parameters': {
                            'missing_values': np.nan,
                            'strategy': 'mean',
                        }
                    },
                    {
                        'uid': '1',
                        'parents': ['0'],
                        'name': 'etimator',
                        'clazz': 'sklearn.ensemble.IsolationForest',
                        'method': 'predict',
                        'hyper_parameters': {}
                    },
                ],
                'sources': {
                    '0': ['NUMERIC']
                }
            }
        ),
    ]
)
def test_train_unsupervised_anomaly_detection(train_dataset: str, model_dict: dict):
    # arrange
    dataset_path = get_test_file(train_dataset)
    reader = DataSetReader()
    model = ModelMetaInfo.from_dict(model_dict)

    # Act
    dataset = reader.read(dataset_path)
    model, _ = train_single(dataset, model, model_type=ModelType.ANOMALY_DETECTION, random_state=123)
    result = model.predict(dataset.get_df())

    # Assert
    assert result.shape[1] == 1
    assert set(result[result.columns[0]].unique()) == {1, -1}