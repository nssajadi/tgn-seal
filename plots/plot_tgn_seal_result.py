import pickle

import matplotlib.pyplot as plt
import numpy as np

DATASET = "email1"


def read_file(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def smooth(y, box_pts=5):
    """Simple moving average smoothing"""
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


models = [
    {"file_path": f"./results/dyrep-rnn-{DATASET}.pkl", "name": "DyRep"},
    {"file_path": f"./results/jodie-rnn-{DATASET}.pkl", "name": "Jodie"},
    {"file_path": f"./results/tgn-id-{DATASET}.pkl", "name": "TGN-id"},
    {"file_path": f"./results/tgn-no-mem-{DATASET}.pkl", "name": "TGN-no-mem"},
    {"file_path": f"./results/tgn-time-{DATASET}.pkl", "name": "TGN-time"},
    {"file_path": f"./results/tgn-seal-{DATASET}-2h.pkl", "name": "TGN-seal-2h"},
]


def plot_metric(metric_key, ylabel, title, smooth_window=5):
    plt.figure(figsize=(10, 6))
    for m in models:
        res = read_file(m["file_path"])
        values = res[metric_key]
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=m["name"], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Plot 1 — Validation AP (Seen Nodes)
plot_metric("val_aps", "Validation AP", "Validation AP (Seen Nodes)")

# Plot 2 — Validation AP (New Nodes)
plot_metric("new_nodes_val_aps", "Validation AP", "Validation AP (New Nodes)")

# Plot 3 — Training Loss
plot_metric("train_losses", "Training Loss", "Training Loss Comparison")
