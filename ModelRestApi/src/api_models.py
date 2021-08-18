from enum import Enum
from typing import List

from pydantic import BaseModel, validator


class PredictRequest(BaseModel):
    """
    Data class for incoming requests
    """
    data: str
    model: str

    @validator("data", "model")
    def check_text(cls, v):
        if not v.strip():
            raise ValueError(f"data / model must not be empty")
        return v


class PredictRequestType(BaseModel):
    """
    Data class for incoming requests
    """
    data: str
    model: str
    type: str

    @validator("data", "model", "type")
    def check_text(cls, v):
        if not v.strip():
            raise ValueError(f"data / model must not be empty")
        return v

    @validator("type")
    def check_category(cls, v):
        if not ModelType.has_key_value(v):
            raise ValueError(f"type {v} is not in the list of possible options: {ModelType.str_key_values()}")
        return v


class PredictionItem(BaseModel):
    """
    Single class that represents a class with it's corresponding prediction
    """
    name: str
    probability: float


class PredictProbaResponse(BaseModel):
    """
    Response class for predictions
    """
    predictions: List[PredictionItem] = []


class ModelType(Enum):
    """
    Enum class to represent different types of ml models
    """
    style = 1
    allergens = 2
    allergens_single = 3

    @classmethod
    def has_int_value(cls, value: int):
        return value in cls._value2member_map_

    @classmethod
    def has_key_value(cls, value: str):
        return value in cls._member_names_

    @classmethod
    def str_key_values(cls):
        return ",".join(cls._member_names_)
