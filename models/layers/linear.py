import tensorflow as tf
import numpy as np


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
        if self.__dropout is not None:
            self.__dropout = tf.constant(self.__dropout, dtype=tf.float32)
            self.__weights = tf.where(
                tf.less(tf.random.uniform(dtype=tf.float32), self.__dropout),
                tf.constant(0.0, dtype=tf.float32),
                self.__weights)
        return tf.add(tf.matmul(inputs, self.__weights), self.__bias)

    @property  # 属性 修饰符
    def input_size(self):
        return self.__input_size

    @property
    def output_size(self):
        return self.__output_size

    @property
    def dropout(self):
        return self.__dropout


if __name__ == "__main__":
    linear = Linear(10, 20)
    result = linear(inputs=tf.convert_to_tensor([[1, 2, 1, 2, 45, 2, 1, 34, 2, 4.0]], dtype=tf.float32))
    print(result)
    pass
