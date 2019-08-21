# -*- encoding: utf-8 -*-
#
# Copyright (c) 2016 Chronos AS
#
# Authors: Fredrik Stormo, Stefan Remman
# Contact: kjetil.karlsen@chronosit.no

import numpy as np

def exponential_decay(n_zero, steps, decay_rate=1.0):
  return [n_zero * np.exp(-decay_rate * t) for t in range(1, steps + 1)]

def exponential_growth(n_zero, steps, growth_rate=1.0):
  return [n_zero * (1 + growth_rate)**t for t in range(1, steps + 1)]

def variance(array):
  to_array(array)
  average = np.average(array)
  variance = 0
  for el in array.tolist():
    variance += (average - el)**2
  return variance/len(array.tolist())

def moving_average(x, window_size):
  window = np.ones(int(window_size))/float(window_size)
  return np.convolve(x, window, 'same')

def moving_average_mat(mat, window_size):
  # Calculate mav for each column in matrix
  window = np.ones(int(window_size))/float(window_size)
  convolutions = []
  for el in mat.T:
    convolution = np.convolve(el, window, 'same')
    convolutions.append(convolution)
  return np.array(convolutions).T
