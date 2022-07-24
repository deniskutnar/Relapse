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
#pet_dirs = glob("/home/denis/samba_share/katrins_data/*/Processed/PET_PET*.nii.gz")
#gtv_dirs = glob("/home/denis/samba_share/katrins_data/*/Processed/GTV*.nii.gz")
#relapse_dirs = glob("/home/denis/samba_share/katrins_data/*/Processed/Relapse*.nii.gz")

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

print(len(ct_dirs))
print(len(pet_dirs))
print(len(gtv_dirs))
print(len(relapse_dirs))




### Split: 23 : 7 : 7

