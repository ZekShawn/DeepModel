

class Feature(object):

    def __init__(self,
                 name=None,
                 val_type=None,
                 tf_type=None,
                 seq_dim=None,
                 seq_len=None,
                 padding=None,
                 is_fixed=False,
                 is_id=False,
                 max_val=None,
                 min_val=None):
        """

        :param name: 特征名称
        :param val_type: 特征类型
        :param tf_type: tensor类型
        :param seq_dim: 序列维度数
        :param seq_len: 序列最大长度
        :param padding: 填充值
        :param is_fixed: 是否是固定值
        :param is_id: 是否是ID类
        """
        super(Feature, self).__init__()
        self.name = name
        self.val_type = val_type
        self.tf_type = tf_type
        self.seq_dim = seq_dim
        self.seq_len = seq_len
        self.padding = padding
        self.is_fixed = is_fixed
        self.is_id = is_id
        self.max_val = max_val
        self.min_val = min_val


class Config(object):

    def __init__(self):
        self.label_std = 700
        self.label_mean = 2000
        self.label = 'label'

    @property
    def features(self):
        return [
            # Feature(name=, val_type=, tf_type=, seq_dim=, seq_len=, padding=, is_fixed=, is_id=, max_val=, min_val=),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature(),
            Feature()
        ]
