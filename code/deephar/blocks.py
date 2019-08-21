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
        (heatmaps, pose), output = self.block_b(a) # Pose Regression
        return heatmaps, pose, output

class PoseRegressionNoContext(nn.Module):
    def __init__(self):
        super(PoseRegressionNoContext, self).__init__()

        self.softargmax = Softargmax(input_filters=16, output_filters=16, kernel_size=(32,32))
        self.probability = JointProbability(filters=16, kernel_size=(32,32))
        self.softmax = nn.Softmax2d()

    def nr_heatmaps(self):
        return 16

    def forward(self, x):
        pose = self.softargmax(x)
        visibility = self.probability(x)
        if torch.cuda.is_available():
            heatmaps = x.cpu().detach().numpy()
        else:
            heatmaps = x.detach().numpy()

        output = torch.cat((pose, visibility), 2)

        return heatmaps, output.unsqueeze(0)

class ActionPredictionBlock(nn.Module):
    def __init__(self, num_actions, num_filters, last=False):
        super(ActionPredictionBlock, self).__init__()

        self.num_actions = num_actions
        self.last = last

        self.acb1 = ACB(input_filters=num_filters, output_filters=int(num_filters/2), kernel_size=(1,1))
        self.acb2 = ACB(input_filters=int(num_filters / 2), output_filters=num_filters, kernel_size=(3,3))

        self.acb3 = ACB(input_filters=num_filters, output_filters=num_filters, kernel_size=(3,3))
        self.pooling = nn.MaxPool2d(kernel_size=(2,2))
        self.ac = AC(input_filters=int(num_filters / 2), output_filters=num_actions, kernel_size=(3, 3))

        if not last:
            self.acb4 = ACB(input_filters=num_actions, output_filters=num_filters, kernel_size=(3, 3))

    def forward(x):
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
            x = a + b + heatmap

        return x, y


class PoseModel(nn.Module):
    def __init__(self, num_frames, num_joints, num_actions):
        super(PoseModel, self).__init__()

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_actions = num_actions

        # TODO: Add layers for preparation of pose
        # multiply visibility score to input
        # does not change dimensions

        # feature extraction
        # TODO: input_filters=2 seems reasonable. Could be wrong though
        self.cba1 = CBA(input_filters=2, output_filters=8, kernel_size(3,1))
        self.cba2 = CBA(input_filters=8, output_filters=16, kernel_size(3,3))
        self.cba3 = CBA(input_filters=16, output_filters=24, kernel_size(3,5))

        # TODO: Check if 48 is correct
        self.cb1 = CB(input_filters=48, output_filters=56, kernel_size(3,3))
        self.cb2 = CB(input_filters=56, output_filters=32, kernel_size(1,1))
        self.cb3 = CB(input_filters=32, output_filters=56, kernel_size(3,3))

        # TODO: Start with simple max pooling
        self.pooling = nn.MaxPool2d(kernel_size=(2,2))

        self.act_pred1 = ActionPredictionBlock(num_actions)
        self.act_pred2 = ActionPredictionBlock(num_actions)
        self.act_pred3 = ActionPredictionBlock(num_actions)
        self.act_pred4 = ActionPredictionBlock(num_actions, last=True)

    def forward(self, x):
        # TODO: Add preparation

        a = self.cba1(x)
        print("a size", a.size())
        b = self.cba2(x)
        print("b size", b.size())
        c = self.cba3(x)
        print("c size", c.size())
        x = torch.cat((a, b, c), 0) # TODO: 0? Which axis?

        a = self.cb1(x)
        b = self.cb2(x)
        b = self.cb3(b)
        x = torch.cat((a, b), 0) # TODO: 0? Which axis?

        x = self.pooling(x)

        x, y1 = self.act_pred1(x)
        x, y2 = self.act_pred2(x)
        x, y3 = self.act_pred3(x)
        x, y4 = self.act_pred4(x)

        x = torch.cat((y1, y2, y3, y4), 0) # TODO: 0? Which axis?
        # should be something like 16 x 4 or 4 x 16
        print("output pose model size", x.size())

        return x
