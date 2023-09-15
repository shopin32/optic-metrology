import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import chardet
from sklearn.model_selection import train_test_split

from optic_metrology.feature import FeatureType, FeaturesMetainfo


class InmemoryDataSet(object):

    def __init__(
            self,
            file_path: str,
            extension: str,
            df: pd.DataFrame,
            metainfo: FeaturesMetainfo,
            encoding: Optional[str] = None,
    ):
        self._extension = extension
        self._file_path = file_path
        self._df = df
        self._metainfo = metainfo
        self._encoding = encoding
    
    def get_df(self) -> pd.DataFrame:
        return self._df
    
    def get_feature_type(self, name: str) -> FeatureType:
        return self._metainfo[name][0]
    
    @property
    def columns(self):
        return self._metainfo.names
    
    @property
    def data_types(self):
        return self._metainfo.types
    
    def sample(
            self, 
            target_name: str, 
            test_size: float = 0.3,
            feature_names: Optional[List[str]] = None,
            random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not feature_names:
            feature_names = [f for f in self._df.columns if f != target_name]
        X = self.get_predictors(target_name, feature_names=feature_names)
        y = self._df[[target_name]]
        if self.get_feature_type(target_name) == FeatureType.NUMERIC:
            return train_test_split(X, y, test_size=test_size, random_state=random_state) 
        unique_categories = self._df[target_name].unique()
        sampled_data = []
        for cat in unique_categories:
            mask = y[target_name] == cat
            sampled_data.append(
                train_test_split(
                    X[mask], y[mask], test_size=test_size, random_state=random_state
                )
            )
        X_train = pd.concat([entry[0] for entry in sampled_data])
        X_test = pd.concat([entry[1] for entry in sampled_data])
        y_train = pd.concat([entry[2] for entry in sampled_data])
        y_test = pd.concat([entry[3] for entry in sampled_data])
        return X_train, X_test, y_train, y_test


    def get_predictors(self, target_name: str, feature_names: Optional[List[str]] = None):
        if not feature_names:
            feature_names = [f for f in self._df.columns if f != target_name]
        return self._df[feature_names]
        
        
class   DataSetReader(object):

    def read(
        self,
        dataset_path: str,
        encoding: Optional[str] = None,
        detect_encoding: bool = False,
        datetime_column: Optional[str] = None,
    ) -> InmemoryDataSet:
        extension = os.path.splitext(dataset_path)[1]
        if extension == '.xlsx':
            df = pd.read_excel(dataset_path)
        elif extension == '.csv':
            if not encoding and detect_encoding:
                encoding = self._detect_encoding(dataset_path)
            df = pd.read_csv(dataset_path)
        metainfo = FeaturesMetainfo()
        for col in df.columns:
            if np.issubdtype(df.dtypes[col], np.number):
                ftype = FeatureType.NUMERIC
            else:
                ftype = FeatureType.CATEGORICAL
            metainfo.add(col, ftype)
        return InmemoryDataSet(
            dataset_path, extension, df, metainfo, encoding=encoding
        )
    
    def _detect_encoding(self, dataset_path: str) -> str:
        with open(dataset_path, 'rb') as rawdata:
            return chardet.detect(rawdata.read(3048))