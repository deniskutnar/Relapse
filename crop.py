import os
import SimpleITK as sitk
import numpy as np
from glob import glob

def read_image(path):
    img = sitk.ReadImage(path)
    img_as_numpy = sitk.GetArrayFromImage(img)
    return img_as_numpy


CT = read_image('/home/abe/samba_share/katrins_data/18103/ct_from_-218_to_192.nii.gz')
PET = read_image('/home/abe/samba_share/katrins_data/18103/pet_from_-782_to_192.nii.gz')
print(CT.shape)
print(PET.shape)



dirs = glob('/home/abe/samba_share/katrins_data/*/*ct_from*')
print(len(dirs))

