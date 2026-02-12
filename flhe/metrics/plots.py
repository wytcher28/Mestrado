import os
import matplotlib.pyplot as plt
import numpy as np
from .evaluation import flatten_cm

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_metric_curve(xs, ys, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(cm_list, title, outpath):
    cm = flatten_cm(cm_list)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0","1"])
    plt.yticks(tick_marks, ["0","1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
