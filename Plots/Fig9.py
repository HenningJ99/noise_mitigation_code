import matplotlib.pyplot as plt
import numpy as np

mm = 1 / 25.4

data = np.genfromtxt("data/Fig9.txt", skip_footer=3)
data_puj = np.genfromtxt("data/Fig9.txt", skip_header=7, skip_footer=1)
data_puj_ind = np.genfromtxt("data/Fig9.txt", skip_header=9)

labels = ["none", "shape global", "both global", "shape local", "both local", "both 0.02 (g)", "RM 0.1", "RM 0.02", "RM 0.09", "RM 0.07", "RM 0.05", "RM 0.03", "RM 0.01"]
fig, axs = plt.subplots(figsize=(88*mm, 88*mm))

axs.set_xticks(range(len(labels)), labels, rotation=45, ha="right", rotation_mode="anchor")

axs.errorbar(0, (data[:,4][0] + data[:, 4][1]) / 2, np.sqrt(data[:,5][0]**2 + data[:,5][1]**2) / 2, fmt='s', markersize=5, color="red", capsize=2)
axs.errorbar(np.arange(1, len(data[:,0])-1), data[:,4][2:], data[:,5][2:], fmt= 's', markersize=5, color="red", capsize=2)
axs.errorbar(np.arange(6, 8), data_puj[:, 4], data_puj[:, 5], fmt="^", markersize=5, color="blue", capsize=2, alpha=0.5)
axs.errorbar(np.arange(6, 8), data_puj[:, 16], data_puj[:, 17], fmt="s", markersize=5, color="red", capsize=2)

axs.set_ylabel("Multiplicative Bias $\mu$")
plt.axvline(x=5.5, color='blue', linestyle='--', alpha=0.6)
plt.text(5.1, -0.03, "Fit method", rotation=90, alpha=0.6, fontsize="small")
plt.text(5.6, -0.03, "Response method", rotation=90, alpha=0.6, fontsize="small")

fig.savefig("Fig9.pdf", dpi=300, bbox_inches="tight")

