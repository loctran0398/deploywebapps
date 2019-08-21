# -*- encoding: utf-8 -*-
#
# Copyright (c) 2016 Chronos AS
#
# Authors: Fredrik Stormo, Stefan Remman
# Contact: kjetil.karlsen@chronosit.no

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import copy
import pandas as pd
from d2o_common.legacy_lib.utils import logger as log
from d2o_common.legacy_lib.utils.handlers import ExceptionHandler

class Scaler:
  """ Simple scaler using scikit-learn with stadard fit() and transform() functionality """
  def __init__(self, method='standard'):
    methods = ['min_max', 'standard']
    if (method.lower().strip() not in methods):
      raise Exception("Unrecognized method %s" % method)

    self.method = method
    self.scaler = None
    self.scaler_fitted = False

  def fit(self, df, **kwargs):
    if (self.method == 'min_max'):
      low = kwargs.get('low',-1)
      high = kwargs.get('high',1)
      self.scaler = MinMaxScaler([low,high])
    elif (self.method == 'standard'):
      self.scaler = StandardScaler()

    X = df.as_matrix()
    self.scaler.fit(X)
    self.scaler_fitted = True

    return self

  def transform(self, df):
    X = df.as_matrix()
    Xs = self.scaler.transform(X)

    ndf = pd.DataFrame(Xs)
    ndf.columns = df.columns
    ndf.index = df.index

    return ndf

  def inverse_transform(self, df):
    Xs = df.as_matrix()

    X = self.scaler.inverse_transform(Xs)
    ndf = pd.DataFrame(X)
    ndf.columns = df.columns
    ndf.index = df.index

    return ndf
