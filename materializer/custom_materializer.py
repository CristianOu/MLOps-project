import os
import pickle
from typing import Any, Type, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME = "CustomerSatisfactionEnvironment"


class cs_materializer(BaseMaterializer):
    """
    Custom materializer for the Customer Satisfaction Project
    """

    ASSOCIATED_TYPES = (
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        CatBoostRegressor,
        RandomForestRegressor,
        LGBMRegressor,
        XGBRegressor,
    )

    def load(self, data_type: Type[Any]) -> Any:
        """
        It loads the object from the artifact store.

        Args:
            data_type: The type of the object to be loaded
        """
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            obj = pickle.load(fid)
        return obj

    def save(self, obj: Any) -> None:
        """
        It saves the object to the artifact store.

        Args:
            obj: The object to be saved
        """
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(obj, fid)
