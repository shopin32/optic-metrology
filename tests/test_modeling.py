import numpy as np
from sklearn import linear_model
from optic_metrology.feature_impact import compute_feature_impact
from optic_metrology.meta_info import ModelMetaInfo
from optic_metrology.reader import DataSetReader
from tests.utils import get_test_file
from optic_metrology.modeling import train, train_single


def test_train_with_model_generation():
    # arrange
    dataset_path = get_test_file('iris_fisher.xlsx')
    target_name = 'Вид ірису'
    reader = DataSetReader()
    dataset = reader.read(dataset_path)

    # Act
    models = train(dataset, target_name, random_state=123422)

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
    models = train(dataset, target_name, random_state=123422)

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
    model, _ = train_single(dataset, target_name, model, 123)
    result = model.predict(dataset.get_df(), show_probs=True)
    feature_impact = compute_feature_impact(dataset, model)

    # Assert
    assert result.shape[1] == 3
