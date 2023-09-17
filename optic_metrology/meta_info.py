from enum import Enum
from typing import Any, Dict, Optional, Union, List

import numpy as np
import pandas as pd

from optic_metrology.feature import FeatureType

class ModelType(Enum):
    REGRESSION = 1
    BINARY = 2
    MULTICLASS = 3
    MULTILABEL = 4

class VertexMetaInfo(object):
    def __init__(
        self, 
        uid: str, 
        name: str, 
        clazz: str,
        method: str, 
        hyper_parameters: dict,
    ) -> None:
        self.uid = uid
        self.name = name
        self.clazz = clazz
        self.parents: List[VertexMetaInfo] = []
        self.children: List[VertexMetaInfo] = []
        self.hyper_parameters = hyper_parameters
        self.method = method
    

    @property
    def parent_uids(self):
        return [parent.uid for parent in self.parents]

    @property
    def children_uids(self):
        return [child.uid for child in self.children]
    
    @classmethod
    def from_dict(cls, dict):
        return cls(dict['uid'], dict['name'], dict['clazz'], dict['method'], dict['hyper_parameters'])
    
    def to_dict(self):
        return {
            'uid': self.uid,
            'name': self.name,
            'parents': [parent.uid for parent in self.parents],
            'children': [child.uid for child in self.children],
            'clazz': self.clazz,
            'hyper_parameters': self.hyper_parameters,
        } 
    
class ModelMetaInfo(object):

    def __init__(self) -> None:
        self.roots: List[VertexMetaInfo] = []
        self.vertices_lookup: Dict[str, VertexMetaInfo] = {}
        self.roots_sources: Dict[str, List[FeatureType]] = {}
    
    def get_source(self, uid: str) -> Optional[List[FeatureType]]:
        return self.roots_sources.get(uid)
    
    def add(
        self,
        vertex: VertexMetaInfo, 
        parents_uids: Optional[List[str]] = None, 
        feature_selection: Optional[List[FeatureType]] = None,
    ) -> None:
        self.vertices_lookup[vertex.uid] = vertex
        if feature_selection:
            self.roots.append(vertex)
            self.roots_sources[vertex.uid] = feature_selection
            return
        if not parents_uids:
            raise ValueError("Either feature_selection of parents_uids are required")
        for parent_uid in parents_uids:
            parent_vertex = self.vertices_lookup[parent_uid]
            parent_vertex.children.append(vertex)
            vertex.parents.append(parent_vertex)
    
    @property
    def vertices(self) -> List[VertexMetaInfo]:
        visited = []
        queue = [] + self.roots
        vertices = []
        while queue:
            v = queue.pop(0) 
            vertices.append(v)
            for child in v.children:
                if child.uid not in visited:
                    visited.append(child.uid)
                    queue.append(child)
        return vertices
    
    def to_dict(self):
        return {
            'vertices': [v.to_dict() for v in self.vertices],
            'sources': {uid: [ft.name for ft in ftypes] for uid, ftypes in self.roots_sources.items()}
        }
    
    @classmethod
    def from_dict(cls, model_dict: dict):
        model = cls()
        for vd in model_dict['vertices']:
            vertex = VertexMetaInfo.from_dict(vd)
            feature_selection = model_dict['sources'].get(vertex.uid)
            if feature_selection:
                feature_selection = [FeatureType[fs] for fs in feature_selection]
            parent_uids = vd['parents'] if vd['parents'] else None
            model.add(vertex, parents_uids=parent_uids, feature_selection=feature_selection)
        return model

