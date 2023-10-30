import numpy as np
import tensorflow as tf


class InputLayer(object):

    def __init__(self,
                 category_cols,
                 numeric_cols,
                 feature_size,
                 field_size,
                 emb_size):
        super(InputLayer, self).__init__()
        self.__category_cols = category_cols
        self.__numeric_cols = numeric_cols
        self.__feature_size = feature_size
        self.__field_size = field_size
        self.__emb_size = emb_size

        # 定义 lookup 表
        stddev = np.sqrt(2 / (self.__field_size + self.__emb_size))
        self.__emb_weights = tf.Variable(
            tf.random_normal(shape=(self.__field_size, self.__emb_size), mean=0, stddev=stddev), dtype=tf.float32)
        self.__emb_bias = tf.Variable(
            tf.random_normal(shape=(1, self.__emb_size), mean=0, stddev=stddev), dtype=tf.float32)

    def __call__(self, feat_index, feat_value):
        """

        :param feat_index:
        :param feat_value:
        :return:
        """
        pass
