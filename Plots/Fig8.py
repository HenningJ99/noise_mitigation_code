import matplotlib.pyplot as plt
import numpy as np

mm = 1 / 25.4

data = np.genfromtxt("data/Fig8.txt", skip_footer=2)
data_puj = np.genfromtxt("data/Fig8.txt", skip_header=4)

labels = ["none", "shape", "both", "both 0.02", "RM 0.1", "RM 0.02"]
fig, axs = plt.subplots(figsize=(88*mm, 88*mm))

axs.set_xticks(range(len(labels)), labels, rotation=45, ha="right", rotation_mode="anchor")

axs.errorbar(np.arange(len(data[:,0])), data[:,4], data[:,5], fmt= 's', markersize=5, color="red", capsize=2)
axs.errorbar(np.arange(len(data_puj[:,0])) + len(data[:,0]), data_puj[:, 7], data_puj[: , 8], fmt="^", markersize=5, color="blue", capsize=2, alpha=0.5)
axs.errorbar(np.arange(len(data_puj[:,0])) + len(data[:,0]), data_puj[:, 20], data_puj[: , 21], fmt="s", markersize=5, color="red", capsize=2)

axs.set_ylabel("Multiplicative Bias $\mu$")
plt.axvline(x=3.5, color='blue', linestyle='--', alpha=0.6)
plt.text(3.1, 0.006, "Fit method", rotation=90, alpha=0.6)
plt.text(3.6, 0.006, "Response method", rotation=90, alpha=0.6)

fig.savefig("Fig8.pdf", dpi=300, bbox_inches="tight")

