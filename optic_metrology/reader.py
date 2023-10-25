import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import chardet
from sklearn.model_selection import train_test_split

from optic_metrology.feature import FeatureType, FeaturesMetainfo
from pandas.api.types import is_datetime64_any_dtype as is_datetime


class InmemoryDataSet(object):

    def __init__(
            self,
            df: pd.DataFrame,
            encoding: Optional[str] = None,
            file_path: Optional[str] = None,
            extension: Optional[str] = None,
    ):
        self._extension = extension
        self._file_path = file_path
        self._df = df
        self._encoding = encoding
        self._metainfo = FeaturesMetainfo()
        for col in df.columns:
            if np.issubdtype(df.dtypes[col], np.number):
                ftype = FeatureType.NUMERIC
            elif is_datetime(df[col]):
                ftype = FeatureType.DATE
            else:
                ftype = FeatureType.CATEGORICAL
                df[col] = df[col].fillna('')
            self._metainfo.add(col, ftype)
    
    def get_df(self) -> pd.DataFrame:
        return self._df
    
    def get_feature_type(self, name: str) -> FeatureType:
        return self._metainfo[name][0]
    
    @property
    def columns(self):
        return self._metainfo.names
    
    @property
    def metainfo(self):
        return self._metainfo
    
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
    
    def get_column(self, name: str) -> pd.Series:
        return self._df[name]
        
        
class   DataSetReader(object):

    def read(
        self,
        dataset_path: str,
        encoding: Optional[str] = None,
        detect_encoding: bool = False,
    ) -> InmemoryDataSet:
        extension = os.path.splitext(dataset_path)[1]
        if extension == '.xlsx':
            df = pd.read_excel(dataset_path)
        elif extension == '.csv':
            if not encoding and detect_encoding:
                encoding = self._detect_encoding(dataset_path)
            df = pd.read_csv(dataset_path)
        return InmemoryDataSet(
            df, encoding=encoding, dataset_path=dataset_path, extension=extension, 
        )
    
    def _detect_encoding(self, dataset_path: str) -> str:
        with open(dataset_path, 'rb') as rawdata:
            return chardet.detect(rawdata.read(3048))
