import json
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.sparse import issparse, hstack
from optic_metrology.feature import FeatureType, FeaturesMetainfo
from optic_metrology.meta_info import ModelMetaInfo, ModelType, VertexMetaInfo
from optic_metrology.utils import get_class
import inspect
import cv2


class Model(object):

    def __init__(
        self, 
        model_meta_info: ModelMetaInfo, 
        features_meta_info: FeaturesMetainfo, 
        model_type: ModelType, 
        target_name: str
    ) -> None:
        self._model_meta_info = model_meta_info
        self._model_type = model_type
        self._features_meta_info = features_meta_info
        self._trained_vertices = {}
        self._le = preprocessing.LabelEncoder()
        self._features: Optional[List[str]] = None
        self._target_name = target_name
        cv2.CAP_PROP_FRAME_COUNT
    

    @property
    def class_names(self) -> List[str]:
        return self._le.classes_
    
    @property
    def target_name(self) -> str:
        return self._target_name
    
    @property
    def features(self) -> List[str]:
        return self._features
    
    @property
    def is_classification(self):
        return self._model_type == ModelType.MULTICLASS
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        self._features = X.columns.tolist()
        extra_parameters = dict(kwargs)
        if self.is_classification:
            self._fit_label_encoder(y)
            y = self._encode_labels(y)
            extra_parameters['num_class'] = len(self.class_names)
        vertices = self._model_meta_info.vertices
        data = {}
        for vertex in vertices:
            hyper_parameters = dict(vertex.hyper_parameters)
            hyper_parameters.update(extra_parameters)
            vertex_clazz = get_class(vertex.clazz)
            result = inspect.getfullargspec(vertex_clazz.__init__)
            hyper_parameters = {n: v for n, v in hyper_parameters.items() if n in result.kwonlyargs + result.args}
            vertex_instance = vertex_clazz(**hyper_parameters)
            vertex_input = self._initialize_vertex_input(X, data, vertex)
            if hasattr(vertex_instance, f'fit_{vertex.method}'):
                vertex_output = getattr(vertex_instance, f'fit_{vertex.method}')(vertex_input, y)
            else:
                vertex_instance.fit(vertex_input, y)
                vertex_output = getattr(vertex_instance, f'{vertex.method}')(vertex_input)
            self._trained_vertices[vertex.uid] = vertex_instance
            data[vertex.uid] = vertex_output


    def predict(self, X: pd.DataFrame, show_probs: bool = False) -> pd.DataFrame:
        X = X[self._features]
        vertices = self._model_meta_info.vertices
        data = {}
        for i in range(len(vertices)):
            vertex = vertices[i]
            vertex_instance = self._trained_vertices[vertex.uid]
            vertex_input = self._initialize_vertex_input(X, data, vertex)
            method_name = vertex.method
            if show_probs and i == len(vertices) - 1:
                method_name = 'predict_proba'
            vertex_output = getattr(vertex_instance, method_name)(vertex_input)
            data[vertex.uid] = vertex_output
        result = vertex_output
        if self.is_classification and not show_probs:
            result = self._le.inverse_transform(vertex_output)
            return pd.DataFrame({'{}_Prediction'.format(self._target_name): result}, index=X.index)
        if self.is_classification and show_probs:
            return pd.DataFrame(data=result, columns=self.class_names)
        return pd.DataFrame({'{}_Prediction'.format(self._target_name): result}, index=X.index)
    
    def score(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> float:
        X = X[self._features]
        vertices = self._model_meta_info.vertices
        y = self._encode_labels(y)
        data = {}
        for i in range(len(vertices)):
            vertex = vertices[i]
            vertex_instance = self._trained_vertices[vertex.uid]
            vertex_input = self._initialize_vertex_input(X, data, vertex)
            method_name = vertex.method
            args = [vertex_input]
            if i == len(vertices) - 1:
                method_name = 'score'
                args += [y]
            vertex_output = getattr(vertex_instance, method_name)(*args)
            data[vertex.uid] = vertex_output
        return vertex_output
    
    def _fit_label_encoder(self, y: pd.DataFrame):
        self._le.fit(y)
    
    def _encode_labels(self, y: pd.DataFrame):
        return self._le.transform(y)
    
    def _initialize_vertex_input(
        self, X: pd.DataFrame, 
        data: Dict[str, np.ndarray], 
        vertex: VertexMetaInfo,
    ):
        if not vertex.parents:
            return self._select_features(X, self._model_meta_info.get_source(vertex.uid))
        return self._stack(data, vertex.parent_uids)
        
    
    def _select_features(self, X: pd.DataFrame, feature_types: List[FeatureType]):
        features_to_select = []
        for col in X.columns:
            if self._features_meta_info[col][0] in feature_types:
                features_to_select.append(col)
        return X[features_to_select]
    
    def _stack(self, data: Dict[str, np.ndarray], parents: List[str]):
        is_sparse = [issparse(data[p]) for p in parents]
        if np.any(is_sparse):
            return hstack([data[p] for p in parents])
        return np.hstack([data[p] for p in parents])
    
    def __repr__(self) -> str:
        vertices = self._model_meta_info.vertices
        return '{' + ','.join(['"{}": {}.{}({})'.format(v.uid, v.clazz.split('.')[-1], v.method, v.parent_uids if v.parents else self._model_meta_info.get_source(v.uid)) for v in vertices]) + '}'

    def __str__(self) -> str:
        return json.dumps(self._model_meta_info.to_dict(), indent=4)
        