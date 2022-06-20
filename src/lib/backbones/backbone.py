import torch.nn as nn


class IntermediateOutputsContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.outputs = []

    def flush(self):
        self.outputs = []

    def forward(self, x):
        self.outputs.append(x)
        return x


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate_outputs_container = IntermediateOutputsContainer()
        self.layers = nn.ModuleList()

    @property
    def intermediate_outputs(self):
        return self.intermediate_outputs_container.outputs

    def forward(self, x):
        self.intermediate_outputs_container.flush()
        for layer in self.layers:
            x = layer(x)
        return x