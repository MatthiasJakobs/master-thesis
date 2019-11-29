import matplotlib.pyplot as plt
import numpy as np

from datasets.JHMDBDataset import actions as jhmdb_actions
from datasets.PennActionDataset import actions as penn_actions

def plot_cm(exp_dir, actions):
    cm = np.load(exp_dir + "cm.np.npy")

    cm = cm / cm.astype(np.float).sum(axis=0)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=actions, yticklabels=actions, xlabel="Predicted class", ylabel="Ground truth class")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=5)
    fig.tight_layout()

    plt.savefig(exp_dir + "cm.png", dpi=500)
    plt.close()

exp_dir = "/data/mjakobs/code/master-thesis/code/experiments/eval_e2e/"
plot_cm(exp_dir, jhmdb_actions)

exp_dir = "/data/mjakobs/code/master-thesis/code/experiments/eval_pennaction/"
plot_cm(exp_dir, penn_actions)
