import torch
import numpy as np
from deephar.layers import *

class Stem(nn.Module):

    def __init__(self):
        super(Stem, self).__init__()

        self.cba1 = CBA(input_filters=3, output_filters=32, kernel_size=(3,3), stride=(2,2))
        self.cba2 = CBA(input_filters=32, output_filters=32, kernel_size=(3,3), stride=(1,1))
        self.cba3 = CBA(input_filters=32, output_filters=64, kernel_size=(3,3), stride=(1,1))
        self.cba4 = CBA(input_filters=64, output_filters=96, kernel_size=(3,3), stride=(2,2))

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        self.cba5 = CBA(input_filters=160, output_filters=64, kernel_size=(1,1), stride=(1,1), padding=0)

        self.cb1 = CB(input_filters=64, output_filters=96, kernel_size=(3,3), stride=(1,1))

        self.cba6 = CBA(input_filters=160, output_filters=64, kernel_size=(1,1), stride=(1,1), padding=0)
        self.cba7 = CBA(input_filters=64, output_filters=64, kernel_size=(5,1), stride=(1,1), padding=1)
        self.cba8 = CBA(input_filters=64, output_filters=64, kernel_size=(1,5), stride=(1,1), padding=1)
        self.cb2 = CB(input_filters=64, output_filters=96, kernel_size=(3,3), stride=(1,1))

        self.acb1 = ACB(input_filters=192, output_filters=192, kernel_size=(3,3), stride=(2,2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)

        self.acb2 = ACB(input_filters=384, output_filters=576, kernel_size=(1,1), stride=(1,1), padding=0)
        self.sep_acb1 = Sep_ACB(input_filters=384, output_filters=576, kernel_size=(3,3), stride=(1,1), padding=1)


    def forward(self, x):
        out = self.cba1(x)
        out = self.cba2(out)
        out = self.cba3(out)

        a = self.cba4(out) #96, 61, 61
        b = self.maxpool1(out) #64,61,61
        out = torch.cat((a,b), 1)

        a = self.cba5(out)
        a = self.cb1(a)

        b = self.cba6(out)
        b = self.cba7(b)
        b = self.cba8(b)
        b = self.cb2(b)

        out = torch.cat((a,b), 1)

        a = self.acb1(out)
        b = self.maxpool2(out)

        out = torch.cat((a,b), 1)

        # sep_conv_residual
        a = self.acb2(out)
        b = self.sep_acb1(out)

        out = a + b

        return out

class BlockA(nn.Module):
    def __init__(self):
        super(BlockA, self).__init__()

        self.acb = ACB(input_filters=576, output_filters=288, kernel_size=(1,1), stride=(1,1), padding=0)

        # calculating padding using
        # padding_zeroes = (kernel_size - 1 ) / 2
        self.sacb1 = Residual(Sep_ACB(input_filters=288, output_filters=288, kernel_size=(5,5), stride=(1,1), padding=2))
        self.sacb2 = Residual(Sep_ACB(input_filters=288, output_filters=288, kernel_size=(5,5), stride=(1,1), padding=2))
        self.sacb3 = Residual(Sep_ACB(input_filters=288, output_filters=288, kernel_size=(5,5), stride=(1,1), padding=2))
        self.sacb4 = Residual(Sep_ACB(input_filters=288, output_filters=288, kernel_size=(5,5), stride=(1,1), padding=2))
        self.sacb5 = Residual(Sep_ACB(input_filters=288, output_filters=288, kernel_size=(5,5), stride=(1,1), padding=2))
        self.sacb6 = Residual(Sep_ACB(input_filters=576, output_filters=576, kernel_size=(5,5), stride=(1,1), padding=2))
        self.sacb7 = Sep_ACB(input_filters=288, output_filters=576, kernel_size=(5,5), stride=(1,1), padding=2)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        a = self.maxpool1(x)
        a = self.acb(a)
        a = self.sacb1(a)

        b = self.maxpool2(a)
        b = self.sacb2(b)
        b = self.sacb3(b)
        b = self.sacb4(b)
        b = nn.functional.interpolate(b, scale_factor=2, mode="nearest") # Maybe align_corners needs to be set?

        b = b + self.sacb5(a)
        b = self.sacb7(b)
        b = nn.functional.interpolate(b, scale_factor=2, mode="nearest") # Maybe align_corners needs to be set?

        return b + self.sacb6(x)

class BlockB(nn.Module):
    def __init__(self, Pose_Regression_Module):
        super(BlockB, self).__init__()

        self.prm = Pose_Regression_Module

        self.sacb = Sep_ACB(input_filters=576, output_filters=576, kernel_size=(5,5), stride=(1,1), padding=2)
        self.ac = AC(input_filters=576, output_filters=self.prm.nr_heatmaps(), kernel_size=(1,1), stride=(1,1), padding=0)
        self.acb = ACB(input_filters=self.prm.nr_heatmaps(), output_filters=576, kernel_size=(1,1), stride=(1,1), padding=0)

    def forward(self, x):
        a = self.sacb(x)
        b = self.ac(a)
        c = self.acb(b)

        return self.prm(b), x + a + c

class ReceptionBlock(nn.Module):
    def __init__(self, num_context=0):
        super(ReceptionBlock, self).__init__()

        self.num_context = num_context
        self.block_a = BlockA()
        if self.num_context > 0:
            raise("Not implemented")
        else:
            self.regression = PoseRegressionNoContext()

        self.block_b = BlockB(self.regression)

    def forward(self, x):
        a = self.block_a(x) # Hourglass
        b = self.block_b(a) # Pose Regression
        return b

class PoseRegressionNoContext(nn.Module):
    def __init__(self):
        super(PoseRegressionNoContext, self).__init__()

        self.softargmax = Softargmax(input_filters=16, output_filters=16, kernel_size=(32,32))
        self.probability = JointProbability(filters=16, kernel_size=(32,32))

    def nr_heatmaps(self):
        return 16

    def forward(self, x):
        pose = self.softargmax(x)
        visibility = self.probability(x)
        heatmaps = x

        output = torch.cat((pose, visibility), 2)

        return output.unsqueeze(0)