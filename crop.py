import os
import SimpleITK as sitk
import numpy as np
from glob import glob

#dirs = glob('/home/denis/samba_share/katrins_data/*/*ct_from*')
#print(len(dirs))

im = sitk.ReadImage('/home/denis/samba_share/katrins_data/18103/ct_from_-218_to_192.nii.gz')

def read_image(path):
    img = sitk.ReadImage(path)
    img_as_numpy = sitk.GetArrayFromImage(img)
    return img_as_numpy




CT = read_image('/home/denis/samba_share/katrins_data/18103/ct_from_-218_to_192.nii.gz')
PET = read_image('/home/denis/samba_share/katrins_data/18103/pet_from_-782_to_192.nii.gz')
print(CT.shape)
print("new pet: ", PET.shape)

PET_cropped = PET[:,:, 282:]
print(PET_cropped.shape)
PET_itk = sitk.GetImageFromArray(PET_cropped)
#PET_itk.CopyInformation(im)

sitk.WriteImage(PET_itk, '/home/denis/samba_share/katrins_data/18103/PET_cropped.nii.gz')
print("done")




