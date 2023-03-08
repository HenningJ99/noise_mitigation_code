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

# if type != "gr":
#     # Copy fits files from first folder
#     shutil.copytree(folder1 + "/FITS_org", folder1.split("run")[-2] + "merged_catalogs/FITS_org")

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


# if type != "gr":
#     # Copy and rename the FITS files from the second folder
#     for file in os.listdir(folder2 + '/FITS_org'):
#         if type == "lf":
#             split = file.split("_")
#             scene_index = int(split[1])
#             scene_index += (np.max(obs1["scene_index"]) +1)
#             shutil.copy(os.path.join(folder2 + '/FITS_org', file), folder1.split("run")[-2] + "merged_catalogs/FITS_org/" + f'catalog_{int(scene_index)}_{split[2]}_{split[3]}')
#         elif type == "puj":
#             scene_index = int(file.split("_")[-2])
#             scene_index += (np.max(obs1["scene_index"]) + 1)
#             shutil.copy(os.path.join(folder2 + '/FITS_org', file),
#                         folder1.split("run")[-2] + "merged_catalogs/FITS_org/" + f'catalog_none_pujol_{int(scene_index)}_' + file.split("_")[-1])
#
#     #Remove duplicates in the individual folders
#     os.system('rm -r ' + folder2 + '/FITS_org')
#     os.system('rm -r ' + folder1 + '/FITS_org')

