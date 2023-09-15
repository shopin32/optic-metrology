from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class FeatureType(Enum):
    NUMERIC = 1
    CATEGORICAL = 2
    TEXT = 3
    DATE = 4
    PERCENTAGE = 5
    LENGTH = 6


class FeaturesMetainfo(object):

    def __init__(self):
        self._feature_to_type: Dict[str, FeatureType] = {}
        self._feature_to_format: Dict[str, str] = {}
    
    @property
    def names(self) -> List[str]:
        return list(self._feature_to_type.keys())
    
    @property
    def types(self) -> List[FeatureType]:
        return [self._feature_to_type[f] for f in self.names]

    def add(self, name: str, ftype: FeatureType, conversion_format: Optional[str] = None):
        self._feature_to_type[name] = ftype
        if conversion_format:
            self._feature_to_format[name] = conversion_format

    def __getitem__(self, name: str) -> Tuple[FeatureType, Optional[str]]:
        return self._feature_to_type[name], self._feature_to_format.get(name)