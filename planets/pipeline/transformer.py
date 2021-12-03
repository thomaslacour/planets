#!/usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import RobustScaler, MinMaxScaler


class Nasa:

    COL_TO_DROP = ['hostname', 'discoverymethod', 'st_metratio', 'decstr']

    def __init__(self, data:pd.DataFrame = None):
        self.X = data
        self.imputer = None
        self.scaler = None
        self._pipeline = None

    def make_imputer(self, transformer='simple', **kwargs):
        """ Imputer strategy to fill na value.
                transformer: str or sklearn.impute
        """
        if type(transformer) == str:
            transformer.lower()
            if transformer == 'knn':
                transformer = KNNImputer(**kwargs)
            if transformer == 'simple':
                transformer = SimpleImputer(**kwargs)

        self.imputer = make_column_transformer(
                (transformer, make_column_selector(dtype_exclude=object)),
                remainder='drop',
                )
        return self


    def make_scaler(self):
        """ Scaler for numerical data (real and integer). Columns that are
            integer must endswith 'num' to be scaled with the right
            transformer.
        """
        # find discontinuous features that should have "num" in their name
        col_int  = [ i for i, u in enumerate(list(self.X.columns)) if u.endswith('num') ]
        # find other features
        col_float = [ i for i, u in enumerate(list(self.X.columns)) if not u.endswith('num') ]

        assert len(col_int + col_float) == len(self.X.columns)

        scaler = make_column_transformer(
            (MinMaxScaler(), col_int),
            (RobustScaler(), col_float),
            remainder='passthrough',
        )
        self.scaler = scaler
        return self


    @property
    def pipeline(self):
        """
        """
        if not self.imputer:
            self.make_imputer
        if not self.scaler:
            self.make_scaler
        self._pipeline = make_pipeline(
                self.imputer,
                self.scaler,
                )
        return self._pipeline
