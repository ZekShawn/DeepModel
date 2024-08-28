import numpy as np
import pandas as pd
from tools.format import Dict2Class


class DataLoader(object):

    def __init__(self, args: Dict2Class):
        self.features = args.features
        self.label = args.label

    def __call__(self):
        pass
