import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(MLP, self).__init__(*args, **kwargs)
        self.layers = kwargs.get("layers", None)
        self.activation = kwargs.get("activation", None)
        assert self.layers is not None and self.activation is not None

        self.net = nn.Sequential()
        for i in range(1, len(self.layers)):
            self.net.append(nn.Linear(self.layers[i-1], self.layers[i]))
            if i != len(self.layers)-1:
                self.net.append(self.activation)

    def forward(self, inputs):
        return self.net(inputs)


class FeatureTransformer(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        pass

    def ft_module(self, input_size, hidden_size, inputs):

        def base_net(input_size, hidden_size):
            net = nn.Sequential()
            net.append(nn.Linear(input_size, hidden_size))
            net.append(nn.BatchNorm1d(hidden_size))
            return net
        
        net = base_net(input_size, hidden_size)
        inputs = net(inputs)
        
        def glu_net(input_size, hidden_size, inputs):
            linear_one = nn.Linear(hidden_size, input_size)
            linear_two = nn.Sequential(nn.Linear(hidden_size, input_size), nn.ReLU())
            inputs = linear_one(inputs) * linear_two(inputs)
            return inputs
        
        inputs = glu_net(input_size, hidden_size, inputs)
        return inputs
    
        