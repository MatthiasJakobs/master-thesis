import torch
import numpy as np
from deephar.layers import *
from deephar.utils import spatial_softmax

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
        self.sep_acb1 = Residual_Sep_ACB(input_filters=384, output_filters=576, kernel_size=(3,3), padding=1)


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
        self.sacb1 = Residual_Sep_ACB(input_filters=288, output_filters=288, kernel_size=(5,5), padding=2)
        self.sacb2 = Residual_Sep_ACB(input_filters=288, output_filters=288, kernel_size=(5,5), padding=2)
        self.sacb3 = Residual_Sep_ACB(input_filters=288, output_filters=288, kernel_size=(5,5), padding=2)
        self.sacb4 = Residual_Sep_ACB(input_filters=288, output_filters=288, kernel_size=(5,5), padding=2)
        self.sacb5 = Residual_Sep_ACB(input_filters=288, output_filters=288, kernel_size=(5,5), padding=2)
        self.sacb6 = Residual_Sep_ACB(input_filters=576, output_filters=576, kernel_size=(5,5), padding=2)
        self.sacb7 = Residual_Sep_ACB(input_filters=288, output_filters=576, kernel_size=(5,5), padding=2)

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
        b = nn.functional.interpolate(b, scale_factor=2, mode="nearest")

        b = b + self.sacb5(a)
        b = self.sacb7(b)
        b = nn.functional.interpolate(b, scale_factor=2, mode="nearest")

        out = b + self.sacb6(x)
        return out

class BlockB(nn.Module):
    def __init__(self, Pose_Regression_Module):
        super(BlockB, self).__init__()

        self.prm = Pose_Regression_Module

        self.sacb = Sep_ACB(input_filters=576, output_filters=576, kernel_size=(5,5), stride=(1,1), padding=2)
        self.ac = AC(input_filters=576, output_filters=self.prm.nr_heatmaps, kernel_size=(1,1), stride=(1,1), padding=0)
        self.acb = ACB(input_filters=self.prm.nr_heatmaps, output_filters=576, kernel_size=(1,1), stride=(1,1), padding=0)

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
            self.regression = PoseRegressionWithContext(self.num_context)
        else:
            self.regression = PoseRegressionNoContext()

        self.block_b = BlockB(self.regression)

    def forward(self, x):
        a = self.block_a(x) # Hourglass
        (heatmaps, pose), output = self.block_b(a) # Pose Regression
        return heatmaps, pose, output

class PoseRegressionNoContext(nn.Module):
    def __init__(self):
        super(PoseRegressionNoContext, self).__init__()

        self.softargmax = Softargmax(input_filters=16, output_filters=16, kernel_size=(32,32))
        self.probability = JointProbability(filters=16, kernel_size=(32,32))

    def nr_heatmaps(self):
        return 16

    def forward(self, x):
        after_softmax = nn.Softmax(2)(x.view(len(x), 16, -1)).view_as(x)
        pose = self.softargmax(after_softmax)
        visibility = self.probability(x)

        output = torch.cat((pose, visibility), 2)

        return after_softmax, output.unsqueeze(0)

class PoseRegressionWithContext(nn.Module):
    def __init__(self, num_context=2, num_joints=16, alpha=0.8):
        super(PoseRegressionWithContext, self).__init__()

        self.num_context = num_context
        self.num_joints = num_joints
        self.alpha = alpha

        self.nr_heatmaps = (self.num_context + 1) * self.num_joints

        self.softargmax_joints = Softargmax(input_filters=self.num_joints, output_filters=self.num_joints, kernel_size=(32,32))
        self.softargmax_context = Softargmax(input_filters=(self.nr_heatmaps - self.num_joints), output_filters=(self.nr_heatmaps - self.num_joints), kernel_size=(32,32))

        self.probability_joints = JointProbability(filters=self.num_joints, kernel_size=(32,32))
        self.probability_context = JointProbability(filters=(self.nr_heatmaps - self.num_joints), kernel_size=(32,32))

    def create_context_sum_layer(self):
        context_sum_layer = nn.Linear((self.nr_heatmaps - self.num_joints), self.num_joints, bias=False)

        w = torch.zeros(((self.nr_heatmaps - self.num_joints), self.num_joints), dtype=torch.float32)
        for i in range(0, self.num_joints):
            w[i * self.num_context : (i + 1) * self.num_context, i] = 1

        w = nn.Parameter(w.t(), requires_grad=False)
        context_sum_layer.weight = w

        return context_sum_layer

    def forward(self, x):
        # Input: Heatmaps
        assert x.shape[1:] == (self.nr_heatmaps, 32, 32)

        # Apply softax to generate belief maps
        x = nn.Softmax(2)(x.view(len(x), self.nr_heatmaps, -1)).view_as(x)

        hs = x[:, :self.num_joints]
        hc = x[:, self.num_joints:]

        assert hs.shape[1:] == (self.num_joints, 32, 32)
        assert hc.shape[1:] == (self.num_joints * self.num_context, 32, 32)

        ys = self.softargmax_joints(hs)
        yc = self.softargmax_context(hc)
        pc = self.probability_context(hc)

        assert ys.shape[1:] == (self.num_joints, 2)
        assert yc.shape[1:] == ((self.nr_heatmaps - self.num_joints), 2)
        assert pc.shape[1:] == ((self.nr_heatmaps - self.num_joints), 1)

        visibility = self.probability_joints(hs)

        # Context aggregation
        pxi = yc[:, :, 0].unsqueeze(-1)
        pyi = yc[:, :, 1].unsqueeze(-1)
        
        pxi = pxi * pc 
        pyi = pyi * pc 

        context_sum = self.create_context_sum_layer()

        # since liner layer expects [batch_size, num_features]: reduce 1-dimension
        pxi = torch.squeeze(pxi, -1)
        pyi = torch.squeeze(pyi, -1)
        pc = torch.squeeze(pc, -1)

        pxi_sum = context_sum(pxi)
        pyi_sum = context_sum(pyi)
        pc_sum = context_sum(pc)

        pxi_div = pxi_sum / pc_sum
        pyi_div = pyi_sum / pc_sum

        assert pxi_div.shape[1:] == (self.num_joints,)
        assert pyi_div.shape[1:] == (self.num_joints,)

        context_x = torch.unsqueeze(pxi_div, -1)
        context_y = torch.unsqueeze(pyi_div, -1)

        context_prediction = torch.cat((context_x, context_y), -1)

        assert context_prediction.shape[1:] == (self.num_joints, 2)

        pose = ys * self.alpha + context_prediction * (1 - self.alpha)

        output = torch.cat((pose, visibility), 2)
        return hs, output.unsqueeze(0)
