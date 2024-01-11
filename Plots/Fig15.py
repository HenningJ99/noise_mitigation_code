from astropy.io import ascii, fits
import numpy as np
import matplotlib.pyplot as plt



data = ascii.read("data/Fig15.txt")

data_random = data[:11200]

data_flagship_new = data[11200:]


mean_blending_fraction_fs_new = [np.average(data_flagship_new[data_flagship_new["mag_auto"] == 20.5 + i]
                                            ["blending_fraction"],
                                     weights=data_flagship_new[data_flagship_new["mag_auto"] == 20.5 + i]
                                            ["weight"])
                          for i in range(7)]

mean_blending_fraction_fs_new_kron = [np.average(data_flagship_new[data_flagship_new["mag_auto"] == 20.5 + i]
                                            ["blending_kron"],
                                     weights=data_flagship_new[data_flagship_new["mag_auto"] == 20.5 + i]
                                            ["weight"])
                          for i in range(7)]

mean_blending_fraction_rp = [np.average(data_random[data_random["mag_auto"] == 20.5 + i]["blending_fraction"],
                                     weights=data_random[data_random["mag_auto"] == 20.5 + i]["weight"])
                          for i in range(7)]

mean_blending_fraction_rp_kron = [np.average(data_random[data_random["mag_auto"] == 20.5 + i]["blending_kron"],
                                     weights=data_random[data_random["mag_auto"] == 20.5 + i]["weight"])
                          for i in range(7)]


mm = 1 / 25.4
fig, axs = plt.subplots(figsize=(88*mm, 88*mm))

axs.plot(np.unique(data_flagship_new["mag_auto"])[:-1]+0.5, mean_blending_fraction_fs_new[:-1], "+--", c="C0")
axs.plot(np.unique(data_flagship_new["mag_auto"])[:-1]+0.5, mean_blending_fraction_fs_new_kron[:-1], "+-", label="Flagship",c="C0")

axs.plot(np.unique(data_random["mag_auto"])[:-1]+0.5, mean_blending_fraction_rp[:-1], "+--", c="C1")
axs.plot(np.unique(data_random["mag_auto"])[:-1]+0.5, mean_blending_fraction_rp_kron[:-1], "+-", label="Random positions",c="C1")

axs.legend()
axs.set_xlabel("$mag_\mathrm{auto}$")
axs.set_ylabel("Blending Fraction")
fig.savefig("Fig15.pdf", dpi=300, bbox_inches="tight")

print(mean_blending_fraction_rp[-1])
print(mean_blending_fraction_rp_kron[-1])
print(mean_blending_fraction_fs_new[-1])
print(mean_blending_fraction_fs_new_kron[-1])