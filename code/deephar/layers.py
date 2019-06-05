import torch.nn as nn

class SeparableConv2D(nn.Module):
    def __init__(self, input_filters, output_filters, kernel_size=3):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(input_filters, input_filters, kernel_size=kernel_size, padding=1, groups=input_filters, bias=False)
        self.pointwise = nn.Conv2d(input_filters, output_filters, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def CBA(input_filters=3, output_filters=32, kernel_size=(3,3), stride=(1,1), padding=1):
    return nn.Sequential(
        nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(output_filters, momentum=0.99, eps=0.001),
        nn.ReLU()
    )

def ACB(input_filters=3, output_filters=32, kernel_size=(3,3), stride=(1,1), padding=1):
    return nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(output_filters, momentum=0.99, eps=0.001)
    )

def Sep_ACB(input_filters=3, output_filters=32, kernel_size=(3,3), stride=(1,1), padding=1):
    return nn.Sequential(
        nn.ReLU(),
        SeparableConv2D(input_filters=384, output_filters=576, kernel_size=(3,3)),
        nn.BatchNorm2d(output_filters, momentum=0.99, eps=0.001)
    )

def CB(input_filters=3, output_filters=32, kernel_size=(3,3), stride=(1,1), padding=1):
    return nn.Sequential(
        nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(output_filters, momentum=0.99, eps=0.001)
    )