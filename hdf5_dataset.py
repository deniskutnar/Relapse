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
relapse_dirs
for idx,f in enumerate(ct_dirs):
    base = os.path.basename(f).split('.nii.gz')[0]
    
    print(base)
    pet = os.path.join(os.path.dirname(os.path.dirname(f)), 'imagesTr/'+ pid + '__PT.nii.gz')
    #print(pet)


    ct = os.path.join(os.path.dirname(os.path.dirname(f)), 'imagesTr/'+ pid + '__CT.nii.gz')
    if os.path.exists(pet):
        pet_dirs.append(pet)
    if os.path.exists(ct):
        ct_dirs.append(ct)




### Split: 23 : 7 : 7

