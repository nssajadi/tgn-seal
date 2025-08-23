import pickle
import matplotlib.pyplot as plt

n_runs = 1


def read_file(file_name):
    with open(file_name, "rb") as f:
        d = pickle.load(f)
    f.close()

    return d


file_path = "./results/tgn-seal.pkl"
results = read_file(file_path)

epochs = range(1, len(results['val_aps']) + 1)

# Plot validation APs
plt.figure(figsize=(8, 5))
plt.plot(epochs, results['val_aps'], marker='o', label="Validation AP (Seen nodes)")
plt.plot(epochs, results['new_nodes_val_aps'], marker='s', label="Validation AP (New nodes)")
plt.xlabel("Epoch")
plt.ylabel("Average Precision (AP)")
plt.title("Validation Performance Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, results['train_losses'], marker='o', color="red")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()
