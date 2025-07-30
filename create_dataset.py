import h5py
import pandas as pd
import numpy as np
import os
from xtb import read_xyz, generate_xtb_features

#When restarting the terminal, I need to run:
#export LD_LIBRARY_PATH=/home/nikolai/OrbitAll/tblite/_build:$LD_LIBRARY_PATH

CSV_FILE = "data/TZP_data/TZP/molecules/molecules.csv"
XYZ_DIR = "data/xyzfiles"
OUTPUT_H5 = "multilevel_PBE_M06-2X.hdf5"

LOW_LEVEL = "PBE"
HIGH_LEVEL = "M06-2X"
EV_TO_MHARTREE = 36.7493

TRAIN_LOW_SIZE, TRAIN_HIGH_SIZE, VAL_SIZE, TEST_SIZE = 2000, 200, 200, 200
np.random.seed(42)

#Load data
df = pd.read_csv(CSV_FILE)
df.set_index("index", inplace=True)

#To make sure indices start from 1
print(all_indices[0:10])

# Shuffle & split indices
all_indices = df.index.to_numpy()
np.random.shuffle(all_indices)

train_low_indices = all_indices[:TRAIN_LOW_SIZE]
train_high_indices = all_indices[TRAIN_LOW_SIZE:TRAIN_LOW_SIZE + TRAIN_HIGH_SIZE]
val_indices = all_indices[TRAIN_LOW_SIZE + TRAIN_HIGH_SIZE:TRAIN_LOW_SIZE + TRAIN_HIGH_SIZE + VAL_SIZE]
test_indices = all_indices[TRAIN_LOW_SIZE + TRAIN_HIGH_SIZE + VAL_SIZE:TRAIN_LOW_SIZE + TRAIN_HIGH_SIZE + VAL_SIZE + TEST_SIZE]

splits = {
    "train": train_low_indices,
    "train_high": train_high_indices,
    "val": val_indices,
    "test": test_indices,
}

# ---------------------------
# Function to write to HDF5
# ---------------------------
def write_group(f_h5, group_name, indices):
    split_grp = f_h5.create_group(group_name)
    i = 1
    for idx in indices:
        print(f"[{group_name}] Processing {i}th molecule")
        i += 1

        try:
            row = df.loc[idx]
        except KeyError:
            print(f"Skipping {idx}, not found in CSV")
            continue

        #Check for NaN in DFT energies
        if pd.isna(row[LOW_LEVEL]) or pd.isna(row[HIGH_LEVEL]):
            print(f"Skipping {idx} due to NaN energy")
            continue

        xyz_file = os.path.join(XYZ_DIR, f"dsgdb9nsd_{idx:06d}.xyz")
        if not os.path.exists(xyz_file):
            print(f"Missing XYZ file for {idx}")
            continue

        #Read molecule
        element_numbers, coords = read_xyz(xyz_file)
        charge = 0
        net_spin = 0

        #Run xTB
        try:
            result = generate_xtb_features(
                element_numbers, coords,
                charge=charge, uhf=net_spin,
                option="GFN1-xTB",
                spin_pol=True,
                get_energy=True,
                get_partial_charges=True
            )
        except Exception as e:
            raise ValueError(f"xTB failed for {idx}: {e}")
            continue

        #Create molecule group
        grp_mol = split_grp.create_group(f"dsgdb9nsd_{idx:06d}")
        geo_grp = grp_mol.create_group("0")  #dummy geometry level

        #Store features
        geo_grp.create_dataset("atomic_numbers", data=element_numbers)
        geo_grp.create_dataset("geometry_bohr", data=coords)
        geo_grp.create_dataset("energy_xtb_Ha", data=result["energy"])
        geo_grp.create_dataset("partial_charges", data=result["partial_charges"])
        geo_grp.create_dataset("charge", data=charge)
        geo_grp.create_dataset("net_spin", data=net_spin)

        two_body = geo_grp.create_group("2body")
        for key in ["F_a", "F_b", "P_a", "P_b", "S", "H"]:
            two_body.create_dataset(key, data=result[key])

        #Store DFT energies
        low_level_mHa = row[LOW_LEVEL]*EV_TO_MHARTREE
        high_level_mHa = row[HIGH_LEVEL]*EV_TO_MHARTREE
        geo_grp.create_dataset(f"{LOW_LEVEL}_energy_mHa", data=low_level_mHa)
        geo_grp.create_dataset(f"{HIGH_LEVEL}_energy_mHa", data=high_level_mHa)

with h5py.File(OUTPUT_H5, "w") as f_h5:
    for split_name, indices in splits.items():
        write_group(f_h5, split_name, indices)

print("\nDone, saved to", OUTPUT_H5)