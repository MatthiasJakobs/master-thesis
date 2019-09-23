import torch

from deephar.utils import get_valid_joints 

def elastic_net_loss_paper(y_pred, y_true):
    # note: not the one used in code but the one in the paper

    valid, nr_valid = get_valid_joints(y_true[:, 0, :, :])
    valid_zero = torch.sum(nr_valid == 0)
    nr_valid[nr_valid == 0] = 1 # just to make sure to not divide by zero
    if valid_zero > 0:
        for i in range(100):
            print("WARNING: nr_valid had a zero in it")
    valid = valid.unsqueeze(1)
    valid = valid.expand(-1, y_pred.size()[1], -1, -1)
    y_pred = y_pred * valid
    y_true = y_true * valid

    difference = torch.abs(y_pred - y_true)
    difference_per_axis = torch.sum(difference, (0, 1, 2))
    #print("difference per axis", difference_per_axis)

    l1 = torch.sum(difference, (1, 2, 3))
    l2 = torch.sum(torch.pow(y_pred - y_true, 2), (1, 2, 3))

    final_losses = (l1 + l2) / nr_valid
    loss = torch.sum(final_losses)
    return loss

def categorical_cross_entropy(y_pred, y_true, axis=1):
    # based on keras implementation

    return -torch.sum(y_true * torch.log(y_pred),  (1,2))
