import numpy as np
import pandas as pd


def _mae(predict: np.ndarray, label: np.ndarray):
    return np.mean(np.abs(predict - label))


def _acc(predict: np.ndarray, label: np.ndarray, start: float = 0, end: float = 12):
    return 0.0


def _mse(predict: np.ndarray, label: np.ndarray):
    return 0.0
