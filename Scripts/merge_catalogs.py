from astropy.table import Table, vstack
from astropy.io import ascii
import sys
import numpy as np
import os
import shutil

folder1 = sys.argv[1]
folder2 = sys.argv[2]

type = sys.argv[3] #lf or puj or gr

os.system('mkdir ' + folder1.split("run")[-2] + 'merged_catalogs')
os.system('mkdir ' + folder1.split("run")[-2] + 'merged_catalogs/mergers')

# Create info file with the two merged catalogs
with open(folder1.split("run")[-2] + 'merged_catalogs/info.txt', 'w') as file:
    file.write(f'{folder1} merged with {folder2}')


# Add the two shear catalogs
obs1 = ascii.read(folder1 + "/shear_catalog.dat", fast_reader={'chunk_size': 100 * 1000000})
obs2 = ascii.read(folder2 + "/shear_catalog.dat", fast_reader={'chunk_size': 100 * 1000000})

if type == "gr":
    obs2["galaxy_id"] += (np.max(obs1["galaxy_id"]) +1)
else:
    obs2["scene_index"] += (np.max(obs1["scene_index"]) + 1)

output = vstack([obs1, obs2])

ascii.write(output, folder1.split("run")[-2] + 'merged_catalogs/shear_catalog.dat',
            overwrite=True)

# Add the two input catalogs
inp1 = ascii.read(folder1 + "/input_catalog.dat", fast_reader={'chunk_size': 100 * 1000000})
inp2 = ascii.read(folder2 + "/input_catalog.dat", fast_reader={'chunk_size': 100 * 1000000})

if type == "gr":
    inp2["galaxy_id"] += (np.max(inp1["galaxy_id"]) + 1)
else:
    inp2["scene_index"] += (np.max(inp1["scene_index"]) + 1)

output = vstack([inp1, inp2])

if os.path.isdir(folder1 + "/mergers"):
    files = os.listdir(folder1 + "/mergers")
    for file in files:
        file_name = os.path.join(folder1 + "/mergers", file)
        shutil.move(file_name, folder1.split("run")[-2] + "merged_catalogs/mergers")
    # shutil.move(folder1 + "/mergers", folder1.split("run")[-2] + "merged_catalogs")
    shutil.rmtree(folder1)
else:
    shutil.move(folder1, folder1.split("run")[-2] + "merged_catalogs/mergers")

if os.path.isdir(folder2 + "/mergers"):
    files = os.listdir(folder2 + "/mergers")
    for file in files:
        file_name = os.path.join(folder2 + "/mergers", file)
        shutil.move(file_name, folder1.split("run")[-2] + "merged_catalogs/mergers")
    #shutil.move(folder2 + "/mergers", folder1.split("run")[-2] + "merged_catalogs")
    shutil.rmtree(folder2)
else:
    shutil.move(folder2, folder1.split("run")[-2] + "merged_catalogs/mergers")

ascii.write(output, folder1.split("run")[-2] + 'merged_catalogs/input_catalog.dat',
            overwrite=True)

