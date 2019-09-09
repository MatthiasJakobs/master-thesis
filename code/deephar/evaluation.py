import numpy as np
from deephar.utils import transform_pose, get_valid_joints

def pckh(y_true, y_pred, head_size, distance_threshold=0.5):
    # from paper: only use joints which are valid, i.e. have a minimum value
    valid = get_valid_joints(y_true, need_sum=False)

    head_size = head_size[0].item()

    squares = np.power(y_true - y_pred, 2)
    sums = np.sum(squares, axis=1)
    distances = np.sqrt(sums) / head_size

    valid = np.apply_along_axis(np.all, axis=1, arr=valid).astype(float)

    matches = (distances <= distance_threshold) * valid

    return np.sum(matches) / np.sum(valid)

def pck_upperbody(y_true, y_pred, distance_threshold=0.5):
    upper_body_size_difference = y_true[0] - y_true[1] # distance between neck and belly, because there is no head size given by the dataset
    
    upper_body_size = torch.sqrt(torch.sum(torch.mul(upper_body_size_difference, upper_body_size_difference)))

    squares = torch.mul(y_true - y_pred, y_true - y_pred)
    sums = torch.sum(squares)
    distances = torch.sqrt(sums) / upper_body_size

    matches = (distances <= distance_threshold)

    return torch.sum(matches) / len(y_true)

def eval_pckh_batch(predictions, poses, headsizes, matrices):
    scores_05 = []
    scores_02 = []

    for i, prediction in enumerate(predictions):
        pred_pose = prediction[:, 0:2]
        ground_pose = poses[i, :, 0:2]

        # transform coordinates back to original
        pred_pose = transform_pose(matrices[i], pred_pose, inverse=True)
        ground_pose = transform_pose(matrices[i], ground_pose, inverse=True)

        scores_05.append(pckh(ground_pose, pred_pose, headsizes[i]))
        scores_02.append(pckh(ground_pose, pred_pose, headsizes[i], distance_threshold=0.2))

    return scores_05, scores_02

