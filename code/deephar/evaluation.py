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

def eval_pckh_batch(model, images, poses, headsizes, matrices):
    scores_05 = []
    scores_02 = []

    model.eval()
    predictions = model(images)
    predictions = predictions[-1, :, :, :].squeeze(dim=0)

    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)

    for i, prediction in enumerate(predictions):
        pred_pose = prediction[:, 0:2]
        ground_pose = poses[i, :, 0:2]

        # transform coordinates back to original
        pred_pose = transform_pose(matrices[i], pred_pose, inverse=True)
        ground_pose = transform_pose(matrices[i], ground_pose, inverse=True)

        scores_05.append(pckh(ground_pose, pred_pose, headsizes[i]))
        scores_02.append(pckh(ground_pose, pred_pose, headsizes[i], distance_threshold=0.2))

    return scores_05, scores_02

