import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BatchNormLayer(object):

    def __init__(self):
        pass

    def __call__(self, inputs, multi_head):
        pass


class Linear(object):

    def __init__(self, input_size, output_size, dropout=None):
        """

        :param input_size:
        :param output_size:
        :param dropout:
        """
        super(Linear, self).__init__()

        # 属性读取
        self.__input_size = input_size
        self.__output_size = output_size
        self.__dropout = dropout

        # 定义参数
        glot = np.sqrt(2 / (input_size + output_size))
        self.__weights = tf.Variable(
            np.random.normal(loc=0.0, scale=glot, size=(input_size, output_size)), dtype=tf.float32)
        self.__bias = tf.constant(np.random.standard_normal(output_size), dtype=tf.float32)

    def __call__(self, inputs):
        matmul_out = tf.matmul(inputs, self.__weights)
        if self.__dropout is not None:
            self.__dropout = tf.constant(self.__dropout, dtype=tf.float32)
            matmul_out = tf.where(
                tf.less(
                    tf.random.uniform(shape=(1, self.__output_size), minval=0, maxval=1, dtype=tf.float32),
                    self.__dropout),
                tf.constant(0.0, dtype=tf.float32),
                matmul_out)
        out = tf.add(matmul_out, self.__bias)
        return out


if __name__ == "__main__":
    linear = Linear(input_size=10, output_size=20, dropout=0.3)
    for i in tqdm(range(100000000)):
        input_tensor = tf.constant(np.random.normal(loc=0.0, scale=6, size=(2, 10)), dtype=tf.float32)
        result = linear(inputs=input_tensor)
