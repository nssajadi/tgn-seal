import pickle

import matplotlib.pyplot as plt

DATASET = "email1"


def read_file(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


models = [
    {"file_path": f"./results/dyrep-rnn-{DATASET}.pkl", "name": "DyRep"},
    {"file_path": f"./results/jodie-rnn-{DATASET}.pkl", "name": "Jodie"},
    {"file_path": f"./results/tgn-id-{DATASET}.pkl", "name": "TGN-id"},
    {"file_path": f"./results/tgn-no-mem-{DATASET}.pkl", "name": "TGN-no-mem"},
    {"file_path": f"./results/tgn-time-{DATASET}.pkl", "name": "TGN-time"},
    {"file_path": f"./results/tgn-seal-{DATASET}-2h.pkl", "name": "TGN-seal-2h"},
]

# =========================================
# Plot 1 — Validation AP (Seen Nodes)
# =========================================
plt.figure(figsize=(9, 6))

for m in models:
    res = read_file(m["file_path"])
    epochs = range(1, len(res['val_aps']) + 1)
    plt.plot(epochs, res['val_aps'], label=m["name"])

plt.xlabel("Epoch")
plt.ylabel("Validation AP")
plt.title("Validation AP (Seen Nodes)")
plt.legend()
plt.show()

# =========================================
# Plot 2 — Validation AP (New Nodes)
# =========================================
plt.figure(figsize=(9, 6))

for m in models:
    res = read_file(m["file_path"])
    epochs = range(1, len(res['new_nodes_val_aps']) + 1)
    plt.plot(epochs, res['new_nodes_val_aps'], label=m["name"])

plt.xlabel("Epoch")
plt.ylabel("Validation AP")
plt.title("Validation AP (New Nodes)")
plt.legend()
plt.show()

# =========================================
# Plot 3 — Training Loss
# =========================================
plt.figure(figsize=(9, 6))

for m in models:
    res = read_file(m["file_path"])
    epochs = range(1, len(res['train_losses']) + 1)
    plt.plot(epochs, res['train_losses'], label=m["name"])

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.show()
