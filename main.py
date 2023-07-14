# 内置的一个类：object


class Demo(object):
    """
    显得规范，看起来也方便
    """

    def __init__(self, size, length, age):
        """
        只会执行一次，在定义的时候，适合放属性
        :param size:
        """
        self.size = size
        self.length = length
        self.__age = age
        self.false = True

    def __call__(self, age):
        """
        可以执行多次，隐式调用
        :return:
        """
        print("__call__")
        return self.length

    def init(self, nums=None):
        """
        在被调用的时候执行
        :return:
        """
        return self.__age

    def call(self):
        pass


if __name__ == "__main__":
    """
    一切皆对象，对象
    """
    demo = Demo(1, 2, 3)
    print(demo.init())
    demo.init = 5
    demo.create = 6
    print(demo.init)
    print(demo.create)
    # demo.__age = 19
    # print(demo.__age)
    #
    #
    # false = True
    #
    # a = 2
    # a = 2.2
    # a = "s"
    # a = ''
    # a = [
    #     [1, 2, 3, 4],  # 0
    #     [2, 3],  # 1
    #     None,  # 2
    #     "c"  # 3
    # ]
    # struct *[] a = [];
    # a = [Demo, 1]
    # a[0](1, 2, 3)
    # b = Demo
    # b(2, 3, 4)
    #
    # print(a[1])
    # print(a[-1::-1])  # 左闭右开 1:3 [1, 3)
    # # start: end: step {}
    #
    # a = {
    #     'key1': 1,
    #     'key2': 2,
    #     'key3': 3
    # }
    #
    # price_dict = {
    #     "水果": {
    #         "苹果": [{}, 1],
    #         "菠萝": 3
    #     }
    # }
    # print(a['key1'])
    # print(price_dict["水果"]["苹果"])
    #
    # # 分支
    # if a is None:
    #     pass
    #
    # if a is not None:
    #     pass
    # elif b is None:
    #     pass
    # else:
    #     pass
    #
    # if 1 < a < 3:
    #     pass
    #
    # a, b = b, a
    #
    # a = 1 if 3 > 5 else 0
    #
    # # 循环
    # for i in range(1, 10):
    #     # i: 1, 2, ..., 9
    #     pass
    #
    # range(10) # 0, 1, 2, ..., 9
    #
    # # for 循环
    # for i in [1, 2, 3, 4, 2]:
    #     i: 1, 2, 3, 4, 2
    #     pass
    #
    # len('123') # 3
    # len([1, 2, 3]) # 3
    # len([
    #     [1, 2, 3],
    #     [],
    #     []
    # ]) # 3
    #
    # for i in range(len([1, 2, 3, 4, 2])):
    #     i: 0, 1, 2, 3, 4
    #
    # # while 循环 - 非0 非None，都为True
    # while 1:
    #     break
    #
    # # 取余 %
    # # 列表推导式
    # a = [1 if x % 2 != 0 else 0 for x in range(10)]
    #
    # # 字典推导式 - 键(唯一的) 值
    # a = {}
    # a = dict()
    # a = {x: 1 if x % 2 != 0 else 0 for x in range(10)}
    #
    # a = [] #  [] 也是一个类，一个内置的类，包含很多函数
    # a = list()  # 显式定义类
    # for x in range(10):
    #     if x % 2 != 0:
    #         a.append(1) # 追加
    #     else:
    #         a.append(0)
    #
    # [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
