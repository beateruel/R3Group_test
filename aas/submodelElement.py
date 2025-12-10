from typing import List, Any
from dataclasses import dataclass

# Classes and functions to handle the submodel element structure for the AAS api

@dataclass
class ModelType:
    name: str

    @staticmethod
    def from_dict(obj: Any) -> 'ModelType':
        if obj is None or not isinstance(obj, dict):
            raise ValueError("ModelType must be a dictionary with a 'name' key.")
        _name = str(obj.get("name"))
        return ModelType(_name)

@dataclass
class Value:
    idShort: str
    modelType: ModelType
    kind: str
    value: List['Value']
    mimeType: str

    @staticmethod
    def from_dict(obj: Any) -> 'Value':
        if obj is None or not isinstance(obj, dict):
            raise ValueError("Value must be a dictionary.")
        _idShort = str(obj.get("idShort", ""))
        _modelType = ModelType.from_dict(obj.get("modelType"))
        _kind = str(obj.get("kind", ""))
        _value = [Value.from_dict(y) for y in obj.get("value", [])] if isinstance(obj.get("value"), list) else []
        _mimeType = str(obj.get("mimeType", ""))
        return Value(_idShort, _modelType, _kind, _value, _mimeType)

@dataclass
class SubmodelElement:
    idShort: str
    modelType: ModelType
    kind: str
    value: List[Value]

    @staticmethod
    def from_dict(obj: Any) -> 'SubmodelElement':
        if obj is None or not isinstance(obj, dict):
            raise ValueError("Root must be a dictionary.")
        _idShort = str(obj.get("idShort", ""))
        _modelType = ModelType.from_dict(obj.get("modelType"))
        _kind = str(obj.get("kind", ""))
        _value = [Value.from_dict(y) for y in obj.get("value", [])] if isinstance(obj.get("value"), list) else []
        return SubmodelElement(_idShort, _modelType, _kind, _value)