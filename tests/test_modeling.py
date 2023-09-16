import numpy as np
from sklearn import linear_model
from optic_metrology.meta_info import ModelMetaInfo
from optic_metrology.reader import DataSetReader
from tests.utils import get_test_file
from optic_metrology.modeling import train, train_single


def test_iris():
    # arrange
    dataset_path = get_test_file('iris_fisher.xlsx')
    target_name = 'Вид ірису'

    # Act
    model = train(dataset_path, target_name, random_state=123422)

    # Assert
    assert isinstance(model, linear_model.SGDClassifier)


def test_meta_info():
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
    model, _ = train_single(dataset_path, target_name, model, 123)
    result = model.predict(dataset.get_df())

    print()
