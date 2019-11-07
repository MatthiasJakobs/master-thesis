import torch
import numpy as np
from deephar.layers import *
from deephar.utils import spatial_softmax

import time

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
        #print("size beginning of stem", torch.cuda.max_memory_allocated() / 1024 / 1024)
        out = self.cba1(x)
        # for layer in self.cba1:
        #     try:
        #         print(layer.weight.size())
        #     except AttributeError:
        #         print(layer, "has no weights")
        #
        # print("output size of stem object", out.nelement() * out.element_size() / 1024 / 1024)
        # print("size after grad calc", torch.cuda.max_memory_allocated() / 1024 / 1024)
        out = self.cba2(out)
        out = self.cba3(out)

        a = self.cba4(out) #96, 61, 61
        b = self.maxpool1(out) #64,61,61
        out = torch.cat((a,b), 1)
        # del a
        # del b
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        a = self.cba5(out)
        a = self.cb1(a)

        b = self.cba6(out)
        b = self.cba7(b)
        b = self.cba8(b)
        b = self.cb2(b)

        out = torch.cat((a,b), 1)
        # del a
        # del b
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        a = self.acb1(out)
        b = self.maxpool2(out)

        out = torch.cat((a,b), 1)
        # del a
        # del b
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        # sep_conv_residual
        a = self.acb2(out)
        # print("size before sepconf", torch.cuda.max_memory_allocated() / 1024 / 1024)
        b = self.sep_acb1(out)
        # print("size after sepconf", torch.cuda.max_memory_allocated() / 1024 / 1024)
        # print(torch.cuda.memory_allocated() / 1024 / 1024)
        # time.sleep(1000)

        out = a + b
        # del a
        # del b
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #
        # print("output size of stem object", out.nelement() * out.element_size() / 1024 / 1024)
        # print(out.shape)
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
        # del a
        # del b
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
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

        self.nr_heatmaps = 16

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

        self.context_sum = self.create_context_sum_layer()

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

        # since liner layer expects [batch_size, num_features]: reduce 1-dimension
        pxi = torch.squeeze(pxi, -1)
        pyi = torch.squeeze(pyi, -1)
        pc = torch.squeeze(pc, -1)

        pxi_sum = self.context_sum(pxi)
        pyi_sum = self.context_sum(pyi)
        pc_sum = self.context_sum(pc)

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

class ActionPredictionBlock(nn.Module):
    def __init__(self, num_actions, num_filters, last=False):
        super(ActionPredictionBlock, self).__init__()

        self.num_actions = num_actions
        self.last = last

        self.acb1 = ACB(input_filters=num_filters, output_filters=int(num_filters/2), kernel_size=(1,1), padding=(0,0))
        self.acb2 = ACB(input_filters=int(num_filters / 2), output_filters=num_filters, kernel_size=(3,3))

        self.acb3 = ACB(input_filters=num_filters, output_filters=num_filters, kernel_size=(3,3))
        self.pooling = MaxMinPooling(kernel_size=(2,2))
        self.ac = AC(input_filters=num_filters, output_filters=num_actions, kernel_size=(3,3))

        if not last:
            self.acb4 = ACB(input_filters=num_actions, output_filters=num_filters, kernel_size=(3,3))

    def forward(self, x):
        a = x
        x = self.acb1(x)
        x = self.acb2(x)
        x = x + a

        a = x
        b = self.acb3(x)
        x = self.pooling(b)
        heatmaps = self.ac(x)
        y = heatmaps

        if not self.last:
            heatmaps = nn.functional.interpolate(heatmaps, scale_factor=2, mode="nearest") # Maybe align_corners needs to be set?
            heatmaps = self.acb4(heatmaps)
            x = a + b + heatmaps
        return x, y

class PoseModelTimeSeries(nn.Module):
    def __init__(self, num_frames, num_joints, num_actions):
        super(PoseModelTimeSeries, self).__init__()

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_actions = num_actions

        # feature extraction
        self.cba1 = CBA(input_filters=2, output_filters=16, kernel_size=(1,4), padding=(1,0))
        self.cba2 = CBA(input_filters=16, output_filters=32, kernel_size=(1,4))
        self.cba3 = CBA(input_filters=32, output_filters=48, kernel_size=(3,3))
        self.cba4 = CBA(input_filters=48, output_filters=56, kernel_size=(3,3))

        self.pooling = MaxMinPooling(kernel_size=(2,2))

        self.act_pred1 = ActionPredictionBlock(num_actions, 112) # TODO: Kind of a magic number
        self.act_pred2 = ActionPredictionBlock(num_actions, 112)
        self.act_pred3 = ActionPredictionBlock(num_actions, 112)
        self.act_pred4 = ActionPredictionBlock(num_actions, 112, last=True)

    def forward(self, x, p):
        # prob = p[:, :, :, 2]
        # prob = prob.unsqueeze(-1)
        # prob = prob.expand(-1, -1, -1, 2)
        # prob = prob.permute(0, 3, 1, 2)

        # x = x * prob # I dont really know why they do that

        x = self.cba1(x)
        x = self.cba2(x)
        x = self.cba3(x)

        x = self.pooling(x)

        x, y1 = self.act_pred1(x)
        x, y2 = self.act_pred2(x)
        x, y3 = self.act_pred3(x)
        _, y4 = self.act_pred4(x)

        return [y1, y2, y3, y4]


class PoseModel(nn.Module):
    def __init__(self, num_frames, num_joints, num_actions):
        super(PoseModel, self).__init__()

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_actions = num_actions

        # feature extraction
        self.cba1 = CBA(input_filters=2, output_filters=8, kernel_size=(3,1), padding=(1,0))
        self.cba2 = CBA(input_filters=2, output_filters=16, kernel_size=(3,3))
        self.cba3 = CBA(input_filters=2, output_filters=24, kernel_size=(3,5), padding=(1,2))

        self.cb1 = CB(input_filters=48, output_filters=56, kernel_size=(3,3))
        self.cb2 = CB(input_filters=48, output_filters=32, kernel_size=(1,1), padding=(0,0))
        self.cb3 = CB(input_filters=32, output_filters=56, kernel_size=(3,3))

        self.pooling = MaxMinPooling(kernel_size=(2,2))

        self.act_pred1 = ActionPredictionBlock(num_actions, 112) # TODO: Kind of a magic number
        self.act_pred2 = ActionPredictionBlock(num_actions, 112)
        self.act_pred3 = ActionPredictionBlock(num_actions, 112)
        self.act_pred4 = ActionPredictionBlock(num_actions, 112, last=True)

    def forward(self, x, p, use_timedistributed=False):
        if use_timedistributed:
            prob = p[:, :, :, 2]
            prob = prob.unsqueeze(-1)
            prob = prob.expand(-1, -1, -1, 2)
            prob = prob.permute(0, 3, 1, 2)
        else:
            prob = p[:, :, 2]
            prob = prob.unsqueeze(-1)
            prob = prob.expand(-1, -1, 2)
            prob = prob.permute(2, 0, 1)

        x = x * prob # I dont really know why they do that

        if not use_timedistributed:
            x = x.unsqueeze(0)
        
        a = self.cba1(x)
        b = self.cba2(x)
        c = self.cba3(x)
        x = torch.cat((a, b, c), 1) # concat on channels

        a = self.cb1(x)
        b = self.cb2(x)
        b = self.cb3(b)

        x = torch.cat((a, b), 1) # concat on channels

        x = self.pooling(x)

        x, y1 = self.act_pred1(x)
        x, y2 = self.act_pred2(x)
        x, y3 = self.act_pred3(x)
        _, y4 = self.act_pred4(x)

        return [y1, y2, y3, y4]

class VisualModel(nn.Module):
    def __init__(self, num_frames, num_joints, num_actions):
        super(VisualModel, self).__init__()

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_actions = num_actions
        self.num_features = 576 # TODO: Can this be derived somehow? what if this changes?

        self.cb = CB(input_filters=self.num_features, output_filters=256, kernel_size=(1,1), padding=(0,0))

        self.pooling = nn.MaxPool2d(kernel_size=(2,2))

        self.act_pred1 = ActionPredictionBlock(num_actions, 256)
        self.act_pred2 = ActionPredictionBlock(num_actions, 256)
        self.act_pred3 = ActionPredictionBlock(num_actions, 256)
        self.act_pred4 = ActionPredictionBlock(num_actions, 256, last=True)

    def forward(self, x, use_timedistributed):
        if not use_timedistributed:
            x = x.unsqueeze(0)
        
        x = self.cb(x)

        x = self.pooling(x)

        x, y1 = self.act_pred1(x)
        x, y2 = self.act_pred2(x)
        x, y3 = self.act_pred3(x)
        _, y4 = self.act_pred4(x)

        return [y1, y2, y3, y4]

class HeatmapWeighting(nn.Module):
    def __init__(self, num_filters):
        super(HeatmapWeighting, self).__init__()

        self.filters = num_filters
        kernel_size = (1,1)

        w1 = torch.ones((num_filters, 1, kernel_size[1], kernel_size[0]), dtype=torch.float32)
        w2 = torch.zeros((num_filters, num_filters, 1, 1), dtype=torch.float32)
        
        for i in range(num_filters):
            w2[i, i, 0, 0] = 1.0

        self.conv = SeparableConv2D(num_filters, num_filters, kernel_size=kernel_size, custom_weights=[nn.Parameter(w1, requires_grad=False), nn.Parameter(w2, requires_grad=False)])

    def forward(self, x):
        return self.conv(x)



