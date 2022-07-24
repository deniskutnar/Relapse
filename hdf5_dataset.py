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
pet_dirs = glob("/home/denis/samba_share/katrins_data/*/Processed/PET_PET*.nii.gz")
gtv_dirs = glob("/home/denis/samba_share/katrins_data/*/Processed/GTV*.nii.gz")
relapse_dirs = glob("/home/denis/samba_share/katrins_data/*/Processed/Relapse*.nii.gz")

print(len(ct_dirs))
print(len(pet_dirs))
print(len(gtv_dirs))
print(len(relapse_dirs))

print(ct_dirs[0])

### Shuffle dirs
ct_dirs = random.shuffle(ct_dirs)
print(ct_dirs[0])
