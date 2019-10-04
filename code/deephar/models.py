import torch.nn as nn

from deephar.blocks import *
from deephar.layers import MaxMinPooling

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
    def __init__(self, num_context=0, standalone=True):
        super(Mpii_4, self).__init__()

        self.stem = Stem()
        self.rec1 = ReceptionBlock(num_context=num_context)
        self.rec2 = ReceptionBlock(num_context=num_context)
        self.rec3 = ReceptionBlock(num_context=num_context)
        self.rec4 = ReceptionBlock(num_context=num_context)

        self.standalone = standalone

    def forward(self, x):
        a = self.stem(x)
        _, pose1 , output1 = self.rec1(a)
        _, pose2 , output2 = self.rec2(output1)
        _, pose3 , output3 = self.rec3(output2)
        heatmaps, pose4 , _ = self.rec4(output3)

        if self.standalone:
            return heatmaps, torch.cat((pose1, pose2, pose3, pose4), 0)
        else:
            return pose4, heatmaps, output1

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
    def __init__(self, num_frames=16, num_joints=16, num_actions=10, use_gt=True, model_path=None, end_to_end=False):
        super(DeepHar, self).__init__()

        self.use_gt = use_gt
        self.end_to_end = end_to_end

        if self.end_to_end:
            assert not self.use_gt

        self.pose_estimator = Mpii_4(num_context=0, standalone=end_to_end)

        if use_gt:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            self.pose_estimator.load_state_dict(torch.load(model_path, map_location=device))
            self.pose_estimator.eval()

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_actions = num_actions

        self.max_min_pooling = MaxMinPooling(kernel_size=(4,4))
        self.softmax = nn.Softmax2d()

        self.pose_model = PoseModel(num_frames, num_joints, num_actions)
        self.visual_model = VisualModel(num_frames, num_joints, num_actions)

        #self.action_predictions = []

    def forward(self, x, train_pose=False):
 
        td_poste_estimator = TimeDistributedPoseEstimation(self.pose_estimator)

        batch_size = len(x)

        if self.end_to_end:
            assert train_pose
        
        if train_pose:
            if self.end_to_end:
                train_heatmaps, train_poses = td_poste_estimator(x)

                heatmaps = train_heatmaps[-1]
                poses = train_poses[-1]
            else:
                poses, heatmaps, features = td_poste_estimator(x)
        else:
            with torch.no_grad():
                poses, heatmaps, features = td_poste_estimator(x)

        nj = poses.size()[2]
        nf = features.size()[2]

        assert nj == self.num_joints

        pose_cube = torch.from_numpy(np.empty((batch_size, self.num_frames, self.num_joints, 2)))
        action_cube = torch.from_numpy(np.empty((batch_size, self.num_frames, self.num_joints, nf)))

        features = features.unsqueeze(2)
        features = features.expand(-1, -1, nj, -1, -1, -1)
        heatmaps = heatmaps.unsqueeze(3)
        heatmaps = heatmaps.expand(-1, -1, -1, nf, -1, -1)

        assert heatmaps.size() == features.size()

        y = features * heatmaps
        y = torch.sum(y, (4, 5))

        action_cube = y
        pose_cube = poses[:, :, :, 0:2]

        pose_cube = pose_cube.permute(0, 3, 1, 2).float()
        action_cube = action_cube.permute(0, 3, 1, 2).float()

        pose_action_predictions = []
        vis_action_predictions = []
        if torch.cuda.is_available():
            pose_cube = pose_cube.to('cuda')
            action_cube = action_cube.to('cuda')

        intermediate_poses = self.pose_model(pose_cube, poses)
        intermediate_vis = self.visual_model(action_cube)

        for y in intermediate_poses:
            y = self.max_min_pooling(y)
            y = self.softmax(y).squeeze(-1).squeeze(-1).unsqueeze(1)

            pose_action_predictions.append(y)

        for y in intermediate_vis:
            y = self.max_min_pooling(y)
            y = self.softmax(y).squeeze(-1).squeeze(-1).unsqueeze(1)

            vis_action_predictions.append(y)

        # TODO: Weighted matrix

        final_vis = intermediate_vis[-1]
        final_pose = intermediate_poses[-1]

        final_output = final_vis + final_pose
        final_output = self.max_min_pooling(final_output)
        final_output = self.softmax(final_output).squeeze(-1).squeeze(-1).unsqueeze(1)

        if self.end_to_end:
            return train_pose, torch.cat(pose_action_predictions, 1), torch.cat(vis_action_predictions, 1), final_output
        else:
            return poses, torch.cat(pose_action_predictions, 1), torch.cat(vis_action_predictions, 1), final_output
