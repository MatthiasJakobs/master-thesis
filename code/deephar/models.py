import torch.nn as nn

from deephar.blocks import *

class Mpii_1(nn.Module):
    def __init__(self, num_context=0):
        super(Mpii_1, self).__init__()

        self.stem = Stem()
        self.rec1 = ReceptionBlock(num_context=num_context)

    def forward(self, x):
        a = self.stem(x)
        heatmaps, output , _ = self.rec1(a)

        return heatmaps, output

class Mpii_2(nn.Module):
    def __init__(self, num_context=0):
        super(Mpii_2, self).__init__()

        self.stem = Stem()
        self.rec1 = ReceptionBlock(num_context=num_context)
        self.rec2 = ReceptionBlock(num_context=num_context)

    def forward(self, x):
        a = self.stem(x)
        _, pose1 , output1 = self.rec1(a)
        heatmaps, pose2 , _ = self.rec2(output1)

        return heatmaps, torch.cat((pose1, pose2), 0)

class Mpii_4(nn.Module):
    def __init__(self, num_context=0):
        super(Mpii_4, self).__init__()

        self.stem = Stem()
        self.rec1 = ReceptionBlock(num_context=num_context)
        self.rec2 = ReceptionBlock(num_context=num_context)
        self.rec3 = ReceptionBlock(num_context=num_context)
        self.rec4 = ReceptionBlock(num_context=num_context)

    def forward(self, x):
        a = self.stem(x)
        _, pose1 , output1 = self.rec1(a)
        _, pose2 , output2 = self.rec2(output1)
        _, pose3 , output3 = self.rec3(output2)
        heatmaps, pose4 , _ = self.rec4(output3)

        return heatmaps, torch.cat((pose1, pose2, pose3, pose4), 0)

class Mpii_8(nn.Module):
    def __init__(self, num_context=0):
        super(Mpii_8, self).__init__()

        self.stem = Stem()
        self.rec1 = ReceptionBlock(num_context=num_context)
        self.rec2 = ReceptionBlock(num_context=num_context)
        self.rec3 = ReceptionBlock(num_context=num_context)
        self.rec4 = ReceptionBlock(num_context=num_context)
        self.rec5 = ReceptionBlock(num_context=num_context)
        self.rec6 = ReceptionBlock(num_context=num_context)
        self.rec7 = ReceptionBlock(num_context=num_context)
        self.rec8 = ReceptionBlock(num_context=num_context)


    def forward(self, x):
        a = self.stem(x)
        _, pose1 , output1 = self.rec1(a)
        _, pose2 , output2 = self.rec2(output1)
        _, pose3 , output3 = self.rec3(output2)
        _, pose4 , output4 = self.rec4(output3)
        _,pose5 , output5 = self.rec5(output4)
        _, pose6 , output6 = self.rec6(output5)
        _, pose7 , output7 = self.rec7(output6)
        heatmaps, pose8 , _ = self.rec8(output7)


        return heatmaps, torch.cat((pose1, pose2, pose3, pose4, pose5, pose6, pose7, pose8), 0)


class DeepHar(nn.Module):
    def __init__(self, num_frames=16, num_joints=16, num_actions=10, use_gt=True):
        super(DeepHar, self).__init__()

        self.use_gt = use_gt
        if not use_gt:
            self.pose_estimator = Mpii_8(num_context=0)

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_actions = num_actions

        self.global_maxmin1 = nn.MaxPool2d(kernel_size=(4,4))
        self.global_maxmin2 = nn.MaxPool2d(kernel_size=(4,4))
        self.softmax = nn.Softmax2d()

        self.pose_model = PoseModel(num_frames, num_joints, num_actions)
        #self.visual_model = VisualModel(num_frames, num_joints, num_actions)

        #self.action_predictions = []

    def forward(self, x, gt):
        if self.use_gt:
            pose = gt
        else:
            # TODO: Get pose from model
            print("not implemented")
        
        action_predictions = []
        intermediate_poses = self.pose_model(pose)
        for y in intermediate_poses:
            y_plus = self.global_maxmin1(y)
            y_minus = self.global_maxmin2(-y)
            y = y_plus - y_minus
            y = self.softmax(y).squeeze(-1).squeeze(-1).unsqueeze(1)

            action_predictions.append(y)

        #y_final = intermediate_poses[-1]
        
        return pose, torch.cat(action_predictions, 1)
