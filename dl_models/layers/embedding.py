import tensorflow as tf
import numpy as np


class EmbeddingLayer(object):

    def __init__(self, embedding_size, feature_size, field_size):
        """

        :param embedding_size:
        :param feature_size: 数值特征个数 + 类别特征属性个数和
        :param field_size: 特征列数
        """
        super(EmbeddingLayer, self).__init__()

        # 定义属性
        self.__embedding_size = embedding_size
        self.__feature_size = feature_size
        self.__field_size = field_size

        # 定义初始化
        glot = np.sqrt(2 / (embedding_size + feature_size))
        self.__embedding_weights = tf.Variable(
            np.random.normal(loc=0, scale=glot, size=(feature_size, embedding_size)), dtype=tf.float32)

        # 定义结果
        self.__embeddings = None

    def __call__(self, feat_index, feat_value):
        """

        :param feat_index:
        :param feat_value:
        :return:
        """
        self.__embeddings = tf.nn.embedding_lookup(self.__embedding_weights, feat_index)
        feat_value = tf.reshape(feat_value, shape=(-1, self.__field_size, 1), name='feat_value')
        self.__embeddings = tf.multiply(self.__embeddings, feat_value, name="embeddings")
        return self.__embeddings


if __name__ == "__main__":
    embedding = EmbeddingLayer(3, 20, 5)
    embeddings = embedding(
        feat_index=tf.convert_to_tensor([[2, 2, 3, 4, 2], [3, 1, 2, 0, 2]], dtype=tf.int64),
        feat_value=tf.convert_to_tensor([[2.2, 2.3, 3, 4, 2], [3.1, 1.4, 2, 0, 2]], dtype=tf.float32)
    )
    print(embeddings)
