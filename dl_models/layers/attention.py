import tensorflow as tf
import numpy as np
import pandas as pd


class SelfAttention(object):

    def __init__(self, q_size, k_size, v_size):
        super(SelfAttention, self).__init__()
        pass

    def __call__(self, q, k, v, mask=None, e=1e-12):
        pass


class MultiHeadAttention(object):

    def __init__(self, head, q, k, v):
        pass

    def __call__(self):
        pass


if __name__ == "__main__":
    print("Hello, world!")
