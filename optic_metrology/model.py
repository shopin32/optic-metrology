from enum import Enum


class ModelType(Enum):
    REGRESSION = 1
    BINARY = 2
    MULTICLASS = 3
    MULTILABEL = 4
