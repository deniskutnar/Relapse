import os
import torch
import torchvision
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import SimpleITK as sitk
from glob import glob

import torch.nn as nn
import nibabel as nib
import shutil
import h5py

import random


ct_dirs = glob("/home/denis/samba_share/katrins_data/*/Processed/CT_CT*.nii.gz")

### Shuffle dirs
random.shuffle(ct_dirs)

pet_dirs = []
gtv_dirs = []
relapse_dirs = []
for f in range(len(ct_dirs)):
    base = ct_dirs[f][:-12]
    pet = base + "PET_PET.nii.gz"
    gtv = base + "GTV.nii.gz"
    relapse = base + "Relapse.nii.gz"

    if os.path.exists(pet):
        pet_dirs.append(pet)
    if os.path.exists(gtv):
        gtv_dirs.append(gtv)
    if os.path.exists(relapse):
        relapse_dirs.append(relapse)


### Split: 23 : 7 : 7 (37)

ct_dirs_train = ct_dirs[:22]
pet_dirs_train = pet_dirs[:22]
gtv_dirs_train = gtv_dirs[:22]
relapse_dirs_train = relapse_dirs[:22]

ct_dirs_val = ct_dirs[22:29]
pet_dirs_val = pet_dirs[22:29]
gtv_dirs_val = gtv_dirs[22:29]
relapse_dirs_val = relapse_dirs[22:29]

ct_dirs_test = ct_dirs[29:]
pet_dirs_test = pet_dirs[29:]
gtv_dirs_test = gtv_dirs[29:]
relapse_dirs_test = relapse_dirs[29:]

print(len(ct_dirs_train))
print(len(ct_dirs_val))
print(len(ct_dirs_test))



