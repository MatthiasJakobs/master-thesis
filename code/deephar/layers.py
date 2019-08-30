import torch.nn as nn
import torch
from deephar.utils import linspace_2d

class SeparableConv2D(nn.Module):
    def __init__(self, input_filters, output_filters, kernel_size=3, padding=1, custom_weights=None):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(input_filters, input_filters, kernel_size=kernel_size, padding=padding, groups=input_filters, bias=False)
        self.pointwise = nn.Conv2d(input_filters, output_filters, kernel_size=1, bias=False)

        if custom_weights is not None:
            self.depthwise.weight = custom_weights[0]
            self.pointwise.weight = custom_weights[1]

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Residual(nn.Module):
    def __init__(self, input_model):
        super(Residual, self).__init__()
        self.input_model = input_model

    def forward(self, x):
        return self.input_model(x) + x

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

def Sep_ACB(input_filters=384, output_filters=576, kernel_size=(3,3), stride=(1,1), padding=1):
    return nn.Sequential(
        nn.ReLU(),
        SeparableConv2D(input_filters=input_filters, output_filters=output_filters, kernel_size=(3,3)),
        nn.BatchNorm2d(output_filters, momentum=0.99, eps=0.001)
    )

def CB(input_filters=3, output_filters=32, kernel_size=(3,3), stride=(1,1), padding=1):
    return nn.Sequential(
        nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(output_filters, momentum=0.99, eps=0.001)
    )

def AC(input_filters=3, output_filters=32, kernel_size=(3,3), stride=(1,1), padding=1):
    return nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    )

class SoftargmaxSimpler(nn.Module):

    def conv_linear_interpolation(self, input_filters, output_filters, kernel_size, dim):
        space = torch.from_numpy(linspace_2d(kernel_size[1], kernel_size[0], dim)).unsqueeze(0).unsqueeze(0).float()
        space = space.expand(-1, input_filters, -1, -1)
        conv = nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size, bias=False)
        with torch.no_grad():
            conv.weight.data = space

        return conv
    
    def __init__(self, input_filters=5, output_filters=5, kernel_size=(3,3)):
        super(SoftargmaxSimpler, self).__init__()

        self.output_filters = output_filters

        self.w_copy = []

        self.xx = self.conv_linear_interpolation(input_filters, input_filters, kernel_size, 0)
        self.xy = self.conv_linear_interpolation(input_filters, input_filters, kernel_size, 1)

    def forward(self, x):
        x_x = self.xx(x)
        x_x = torch.squeeze(x_x, dim=-1)
        x_x = torch.squeeze(x_x, dim=-1)
        
        x_y = self.xy(x)
        x_y = torch.squeeze(x_y, dim=-1)
        x_y = torch.squeeze(x_y, dim=-1)        

        x = torch.cat((x_x, x_y))
        #return x.reshape((-1, self.output_filters, 2))#, self.w_copy
        return x

class Softargmax(nn.Module):

    def conv_linear_interpolation(self, input_filters, output_filters, kernel_size, dim):
        space = linspace_2d(kernel_size[1], kernel_size[0], dim)

        w1 = torch.zeros((input_filters, 1, kernel_size[1], kernel_size[0]), dtype=torch.float32)
        w2 = torch.zeros((input_filters, input_filters, 1, 1), dtype=torch.float32)
        if dim == 1:
            self.w_copy = w1
        
        for i in range(input_filters):
            for idx in range(2):
                if idx == 0:
                    w1[i, 0, :, :] = torch.from_numpy(space)
                else:
                    w2[i, i, 0, 0] = 1.0

        conv = SeparableConv2D(input_filters=input_filters, output_filters=output_filters, kernel_size=kernel_size, padding=0, custom_weights=[nn.Parameter(w1, requires_grad=False), nn.Parameter(w2, requires_grad=False)])

        return conv
    
    def __init__(self, input_filters=5, output_filters=5, kernel_size=(3,3)):
        super(Softargmax, self).__init__()

        self.output_filters = output_filters

        self.w_copy = []

        self.xx = self.conv_linear_interpolation(input_filters, input_filters, kernel_size, 0)
        self.xy = self.conv_linear_interpolation(input_filters, input_filters, kernel_size, 1)

    def forward(self, x):
        x_x = self.xx(x)
        x_x = torch.squeeze(x_x, dim=-1)
        x_x = torch.squeeze(x_x, dim=-1)
        x_x = x_x.reshape((-1, self.output_filters, 1))
        
        x_y = self.xy(x)
        x_y = torch.squeeze(x_y, dim=-1)
        x_y = torch.squeeze(x_y, dim=-1)
        x_y = x_y.reshape((-1, self.output_filters, 1))

        x = torch.cat((x_x, x_y), 2)
        return x

class JointProbability(nn.Module):
    def __init__(self, filters=16, kernel_size=(32,32)):
        super(JointProbability, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.sigmoid(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
  
        return x.reshape((-1, 16, 1))
