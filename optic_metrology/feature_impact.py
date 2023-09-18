
from optic_metrology.model import Model
from optic_metrology.reader import InmemoryDataSet
from sklearn.inspection import permutation_importance

def compute_feature_impact(dataset: InmemoryDataSet, model: Model, **kwargs):
    feature_names = model.features
    X_train, X_val, y_train, y_val = dataset.sample(model.target_name, feature_names=feature_names)

    r = permutation_importance(model, X_val, y_val.values.ravel(), **kwargs)
    result = {}
    for i in r.importances_mean.argsort()[::-1]:
        result[feature_names[i]] = (r.importances_mean[i], r.importances_std[i])
    return result
