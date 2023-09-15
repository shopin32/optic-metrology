from sklearn import linear_model
from tests.utils import get_test_file
from optic_metrology.modeling import train


def test_iris():
    # arrange
    dataset_path = get_test_file('iris_fisher.xlsx')
    target_name = 'Вид ірису'

    # Act
    model = train(dataset_path, target_name, random_state=123422)

    # Assert
    assert isinstance(model, linear_model.SGDClassifier)