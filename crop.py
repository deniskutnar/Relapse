import os
import SimpleITK as sitk
import numpy as np
from glob import glob

def read_image(path):
    img = sitk.ReadImage(path)
    img_as_numpy = sitk.GetArrayFromImage(img)
    return img_as_numpy

ct_dirs = glob('/home/denis/samba_share/katrins_data/*/*ct_from*')
pet_dirs = glob('/home/denis/samba_share/katrins_data/*/*pet_from*')


for f in range (len(ct_dirs)):
    print('"'+ ct_dirs[f]+ '"')




#for f in range(len(ct_dirs)):
  #  name = (ct_dirs[f].rsplit('/', 1)[-1])
   # first = name[8:12]
    #second = name [16:19]
    #print (first)
    #print ("s ",second)





  

  

  




#CT = read_image('/home/denis/samba_share/katrins_data/18103/ct_from_-218_to_192.nii.gz')
#PET = read_image('/home/denis/samba_share/katrins_data/18103/pet_from_-782_to_192.nii.gz')
#print(CT.shape)
#print("new pet: ", PET.shape)

#PET_cropped = PET[:,:, 282:]
#print(PET_cropped.shape)
#PET_itk = sitk.GetImageFromArray(PET_cropped)
#PET_itk.CopyInformation(im)

#sitk.WriteImage(PET_itk, '/home/denis/samba_share/katrins_data/18103/PET_cropped.nii.gz')
#print("done")




