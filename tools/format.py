import json


def json2dict(json_file):
    """
    convert json str to dict
    """
    dic = None
    try:
        line = ""
        with open(json_file, mode="r") as file:
            lines = file.readlines()
            line = line + "".join(lines)
            line = line.replace("\n", "").replace(" ", "")
        dic = json.loads(line)
    except Exception as e:
        print(e)
        dic = {}
    finally:
        return dic


def class2dict(ins, cls):
    dic = {}
    if isinstance(ins, cls):
        for key, value in ins.__dict__.items():
            if isinstance(value, cls):
                dic[key] = class2dict(value, cls)
            else:
                dic[key] = value
    return dic


class Dict2Class(object):
    """
    convert dict to class
    """

    def __init__(self, args: dict) -> None:
        super(Dict2Class, self).__init__()
        assert isinstance(args, dict)
        self.update_attributes(args)

    def update_attributes(self, attr_dict: dict):
        assert isinstance(attr_dict, dict)
        for key, value in attr_dict.items():
            if isinstance(value, dict):
                self.__setattr__(key, Dict2Class(value))
            else:
                self.__setattr__(key, value)

    def set_attribute(self, name: str, value):
        assert isinstance(name, str)
        return self.__setattr__(name, value)

    def get_attribute(self, name: str):
        assert isinstance(name, str)
        return self.__getattribute__(name)
