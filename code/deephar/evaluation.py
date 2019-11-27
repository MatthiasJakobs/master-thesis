import torch
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

    return matches, valid, np.sum(matches) / np.sum(valid)

def pck_bounding_box(y_true, y_pred, distance_meassure, distance_threshold=0.2):
    assert distance_meassure != 0

    valid = get_valid_joints(y_true, need_sum=False)

    initial_distances = y_true - y_pred

    squares = torch.mul(initial_distances, initial_distances)
    sums = torch.sum(squares, (1,))
    distances = torch.sqrt(sums) / distance_meassure

    if torch.cuda.is_available():
        valid = torch.from_numpy(np.apply_along_axis(np.all, axis=1, arr=valid.cpu()).astype(float)).float().to("cuda")
        matches = (distances <= distance_threshold).float().to("cuda") * valid
    else:
        valid = torch.from_numpy(np.apply_along_axis(np.all, axis=1, arr=valid).astype(float)).float()
        matches = (distances <= distance_threshold).float() * valid


    return matches, valid, torch.sum(matches) / torch.sum(valid)

def pck_upperbody(y_true, y_pred, distance_threshold=0.5, compute_upperbody=False):
    if compute_upperbody:
        upper_middle = y_true[12] + (y_true[13] - y_true[12]) / 2.0
        lower_middle = y_true[3] + (y_true[2] - y_true[3]) / 2.0
        upper_body_size_difference = upper_middle - lower_middle
        upper_body_size = torch.sqrt(torch.sum(torch.mul(upper_body_size_difference, upper_body_size_difference)))
        print(upper_body_size)
    else:
        upper_body_size_difference = y_true[8] - y_true[6] # distance between neck and belly, because there is no head size given by the dataset
        upper_body_size = torch.sqrt(torch.sum(torch.mul(upper_body_size_difference, upper_body_size_difference)))

    assert upper_body_size != 0

    valid = get_valid_joints(y_true, need_sum=False)

    initial_distances = y_true - y_pred

    squares = torch.mul(initial_distances, initial_distances)
    sums = torch.sum(squares, (1,))
    distances = torch.sqrt(sums) / upper_body_size

    if torch.cuda.is_available():
        valid = torch.from_numpy(np.apply_along_axis(np.all, axis=1, arr=valid.cpu()).astype(float)).float().to("cuda")
        matches = (distances <= distance_threshold).float().to("cuda") * valid
    else:
        valid = torch.from_numpy(np.apply_along_axis(np.all, axis=1, arr=valid).astype(float)).float()
        matches = (distances <= distance_threshold).float() * valid


    return matches, valid, torch.sum(matches) / torch.sum(valid)

def eval_pcku_batch(predictions, poses, matrices, compute_upperbody=False, return_perjoint=False):
    scores_02 = []
    matches = []
    valids = []

    for i, prediction in enumerate(predictions):

        # transform coordinates back to original
        pred_pose = torch.from_numpy(transform_pose(matrices[i], predictions[i], inverse=True))
        ground_pose = torch.from_numpy(transform_pose(matrices[i], poses[i], inverse=True))

        match, valid, scores = pck_upperbody(ground_pose, pred_pose, distance_threshold=0.2, compute_upperbody=compute_upperbody)
        if torch.cuda.is_available():
            matches.append(match.cpu().numpy())
            valids.append(valid.cpu().numpy())
        else:
            matches.append(match.numpy())
            valids.append(valid.numpy())
        scores_02.append(scores)

    if return_perjoint:
        return matches, valids
    else:
        return scores_02

def eval_pck_batch(predictions, poses, matrices, distance_meassures, threshold=0.2, return_perjoint=False):
    scores_02 = []
    matches = []
    valids = []

    for i, prediction in enumerate(predictions):

        # transform coordinates back to original
        pred_pose = torch.from_numpy(transform_pose(matrices[i], predictions[i], inverse=True))
        ground_pose = torch.from_numpy(transform_pose(matrices[i], poses[i], inverse=True))

        match, valid, scores = pck_bounding_box(ground_pose, pred_pose, distance_meassures[i], distance_threshold=threshold)
        scores_02.append(scores)
        if torch.cuda.is_available():
            matches.append(match.cpu().numpy())
            valids.append(valid.cpu().numpy())
        else:
            valids.append(valid.numpy())
            matches.append(match.numpy())

    if return_perjoint:
        return matches, valids
    else:
        return scores_02


def eval_pckh_batch(predictions, poses, headsizes, matrices, return_perjoint=False):
    scores_05 = []
    scores_02 = []
    matches_05 = []
    matches_02 = []
    valids_02 = []
    valids_05 = []

    for i, prediction in enumerate(predictions):
        pred_pose = prediction[:, 0:2]
        ground_pose = poses[i, :, 0:2]

        # transform coordinates back to original
        pred_pose = transform_pose(matrices[i], pred_pose, inverse=True)
        ground_pose = transform_pose(matrices[i], ground_pose, inverse=True)

        match, valid, scores = pckh(ground_pose, pred_pose, headsizes[i])
        if torch.cuda.is_available():
            matches_05.append(match.cpu().numpy())
            valids_05.append(valid.cpu().numpy())
        else:
            matches_05.append(match.numpy())
            valids_05.append(valid.numpy())
        scores_05.append(scores)
        match, scores = pckh(ground_pose, pred_pose, headsizes[i], distance_threshold=0.2)
        if torch.cuda.is_available():
            matches_02.append(match.cpu().numpy())
            valids_02.append(valid.cpu().numpy())
        else:
            matches_02.append(match.numpy())
            valids_02.append(valid.numpy())
        scores_02.append(scores)

    if return_perjoint:
        return matches_05, matches_02, valids_05, valids_02
    else:
        return scores_05, scores_02

