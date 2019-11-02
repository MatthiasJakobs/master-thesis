import torch.nn as nn

from deephar.blocks import *
from deephar.layers import MaxMinPooling
from deephar.utils import create_heatmap

from skimage.transform import resize

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

        self.blocks = 2

        self.stem = Stem()
        self.rec1 = ReceptionBlock(num_context=num_context)
        self.rec2 = ReceptionBlock(num_context=num_context)

    def forward(self, x):
        a = self.stem(x)
        _, pose1 , output1 = self.rec1(a)
        heatmaps, pose2 , _ = self.rec2(output1)

        return torch.cat((pose1, pose2), 0), pose2, heatmaps, output1

class Mpii_4(nn.Module):
    def __init__(self, num_context=0):
        super(Mpii_4, self).__init__()

        self.blocks = 4

        self.stem = Stem()
        self.rec1 = ReceptionBlock(num_context=num_context)
        self.rec2 = ReceptionBlock(num_context=num_context)
        self.rec3 = ReceptionBlock(num_context=num_context)
        self.rec4 = ReceptionBlock(num_context=num_context)

    def forward(self, x):
        a = self.stem(x)
        #print("overall usage after stem", torch.cuda.max_memory_allocated() / 1024 / 1024)
        _, pose1 , output1 = self.rec1(a)

        del a
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        #print("overall usage after r1", torch.cuda.max_memory_allocated() / 1024 / 1024)

        _, pose2 , output2 = self.rec2(output1)
        #print("overall usage after r2", torch.cuda.max_memory_allocated() / 1024 / 1024)
        _, pose3 , output3 = self.rec3(output2)
        #print("overall usage after r3", torch.cuda.max_memory_allocated() / 1024 / 1024)

        del output2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        #print("overall usage before final", torch.cuda.max_memory_allocated() / 1024 / 1024)

        heatmaps, pose4 , _ = self.rec4(output3)
        del output3
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        #print("overall usage after r4", torch.cuda.max_memory_allocated() / 1024 / 1024)
        return torch.cat((pose1, pose2, pose3, pose4), 0), pose4, heatmaps, output1


class Mpii_8(nn.Module):
    def __init__(self, num_context=0):
        super(Mpii_8, self).__init__()

        self.blocks = 8

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


        return torch.cat((pose1, pose2, pose3, pose4, pose5, pose6, pose7, pose8), 0), pose8, heatmaps, output1


class DeepHar(nn.Module):
    def __init__(self, num_frames=16, num_joints=16, num_actions=10, use_gt=True, model_path=None, alternate_time=False):
        super(DeepHar, self).__init__()

        self.use_gt = use_gt # use pretrained pose estimator
        self.alternate_time = alternate_time
        self.pose_estimator = Mpii_4(num_context=0)

        if use_gt:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            self.pose_estimator.load_state_dict(torch.load(model_path, map_location=device))

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_actions = num_actions

        self.max_min_pooling = MaxMinPooling(kernel_size=(4,4))
        self.softmax = nn.Softmax2d()

        if self.alternate_time:
            self.pose_model = PoseModelTimeSeries(num_frames, num_joints, num_actions)
        else:
            self.pose_model = PoseModel(num_frames, num_joints, num_actions)
        self.visual_model = VisualModel(num_frames, num_joints, num_actions)

        self.heatmap_weighting = HeatmapWeighting(self.num_actions)

        #self.action_predictions = []

    def extract_pose_for_frames(self, x, gt_pose=None):
        train_poses, poses, heatmaps, features = self.pose_estimator(x)
        poses = poses.squeeze(0)
        train_poses = train_poses.permute(1, 0, 2, 3)

        if gt_pose is not None:
            assert gt_pose.shape == (16, 16, 3)
            train_poses = gt_pose.clone()
            train_poses = train_poses.unsqueeze(1)
            train_poses = train_poses.expand(-1, 4, -1, -1)

            heatmaps = torch.FloatTensor(16, 16, 32, 32)
            for i in range(16):
                for o in range(16):
                    pose = gt_pose[i, o]
                    if pose[2] == 0:
                        mean_x = 127
                        mean_y = mean_x
                        cov = 500
                    else:
                        mean_x = pose[0] * 255
                        mean_y = pose[1] * 255
                        cov = 10

                    heatmap = create_heatmap(mean_x, mean_y, cov)
                    heatmap = torch.FloatTensor(resize(heatmap, (32, 32)))
                    heatmaps[i, o] = heatmap

            poses = gt_pose.clone()

        return train_poses, poses, heatmaps, features

    def forward(self, x, finetune=False, gt_pose=None):

        if finetune:
            train_poses, poses, heatmaps, features = self.extract_pose_for_frames(x, gt_pose=gt_pose)
        else:
            with torch.no_grad():
                train_poses, poses, heatmaps, features = self.extract_pose_for_frames(x, gt_pose=gt_pose)

        nj = poses.size()[1]
        nf = features.size()[1]

        assert nj == self.num_joints

        features = features.unsqueeze(1)
        features = features.expand(-1, nj, -1, -1, -1)
        heatmaps = heatmaps.unsqueeze(2)
        heatmaps = heatmaps.expand(-1, -1, nf, -1, -1)

        assert heatmaps.size() == features.size()
        y = features * heatmaps
        y = torch.sum(y, (3, 4))

        del features
        del heatmaps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        action_cube = torch.from_numpy(np.empty((self.num_frames, self.num_joints, nf)))

        action_cube = y
        del y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pose_cube = torch.from_numpy(np.empty((self.num_frames, self.num_joints, 2)))
        pose_cube = poses[:, :, 0:2]

        if self.alternate_time:
            pose_cube = pose_cube.permute(2, 1, 0).float()
        else:
            pose_cube = pose_cube.permute(2, 0, 1).float()

        action_cube = action_cube.permute(2, 0, 1).float()

        pose_action_predictions = []
        vis_action_predictions = []
        if torch.cuda.is_available():
            pose_cube = pose_cube.to('cuda')
            action_cube = action_cube.to('cuda')

        intermediate_poses = self.pose_model(pose_cube, poses)
        intermediate_vis = self.visual_model(action_cube)
        del pose_cube
        del action_cube
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for y in intermediate_poses:
            y = self.max_min_pooling(y)
            y = self.softmax(y).squeeze(-1).squeeze(-1).unsqueeze(1)

            pose_action_predictions.append(y)

        for y in intermediate_vis:
            y = self.max_min_pooling(y)
            y = self.softmax(y).squeeze(-1).squeeze(-1).unsqueeze(1)

            vis_action_predictions.append(y)

        final_vis = intermediate_vis[-1]
        final_pose = intermediate_poses[-1]

        final_vis = self.heatmap_weighting(final_vis)
        final_pose = self.heatmap_weighting(final_pose)

        final_output = final_vis + final_pose
        final_output = self.max_min_pooling(final_output)
        final_output = self.softmax(final_output).squeeze(-1).squeeze(-1).unsqueeze(1)

        return train_poses, poses, torch.cat(pose_action_predictions, 1), torch.cat(vis_action_predictions, 1), final_output
