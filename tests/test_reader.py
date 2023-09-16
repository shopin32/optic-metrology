from optic_metrology.feature import FeatureType
from optic_metrology.reader import DataSetReader
from tests.utils import get_test_file


def test_iris_reader():
    # Arrange
    dataset_path = get_test_file('iris_fisher.xlsx')
    reader = DataSetReader()

    # Act
    dataset = reader.read(dataset_path, detect_encoding=True)

    # Assert
    assert dataset != None
    assert dataset.columns == ['Довжина оцвітини', 'Ширина чашолистка', 'Довжина пелюстки', 'Ширина пелюстки', 'Вид ірису']
    assert dataset.data_types == [FeatureType.NUMERIC, FeatureType.NUMERIC, FeatureType.NUMERIC, FeatureType.NUMERIC, FeatureType.CATEGORICAL]


def test_diabetes_reader():
    # Arrange
    dataset_path = get_test_file('diabetic_data.csv')
    reader = DataSetReader()

    # Act
    dataset = reader.read(dataset_path, detect_encoding=True)
    
    # Assert
    assert dataset != None

