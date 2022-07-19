import os
import SimpleITK as sitk
import numpy as np
from glob import glob
import shutil

def read_image(path):
    img = sitk.ReadImage(path)
    img_as_numpy = sitk.GetArrayFromImage(img)
    return img_as_numpy

ct_dirs = glob('/home/denis/samba_share/katrins_data/*/*ct_from*')
pet_dirs = glob('/home/denis/samba_share/katrins_data/*/*pet_from*')

#gtv_dirs = glob('/home/denis/samba_share/katrins_data/*/GTV*')



OK_dirs = ["/home/denis/samba_share/katrins_data/6747", 
"/home/denis/samba_share/katrins_data/6823",
"/home/denis/samba_share/katrins_data/7729",
"/home/denis/samba_share/katrins_data/7660",
"/home/denis/samba_share/katrins_data/9515",
"/home/denis/samba_share/katrins_data/9649",
"/home/denis/samba_share/katrins_data/9777",
"/home/denis/samba_share/katrins_data/9930",
"/home/denis/samba_share/katrins_data/11386",
"/home/denis/samba_share/katrins_data/17496"]

OneS_dirs = ["/home/denis/samba_share/katrins_data/8935",
"/home/denis/samba_share/katrins_data/9610",
"/home/denis/samba_share/katrins_data/9937",
"/home/denis/samba_share/katrins_data/10033",
"/home/denis/samba_share/katrins_data/10157",
"/home/denis/samba_share/katrins_data/10188",
"/home/denis/samba_share/katrins_data/11061",
"/home/denis/samba_share/katrins_data/11086",
"/home/denis/samba_share/katrins_data/11210",
"/home/denis/samba_share/katrins_data/11576",
"/home/denis/samba_share/katrins_data/11663",
"/home/denis/samba_share/katrins_data/13271",
"/home/denis/samba_share/katrins_data/13282",
"/home/denis/samba_share/katrins_data/13503",
"/home/denis/samba_share/katrins_data/13526",
"/home/denis/samba_share/katrins_data/13576",
"/home/denis/samba_share/katrins_data/13648",
"/home/denis/samba_share/katrins_data/13777",
"/home/denis/samba_share/katrins_data/14034",
"/home/denis/samba_share/katrins_data/14049",
"/home/denis/samba_share/katrins_data/14401",
"/home/denis/samba_share/katrins_data/14800",
"/home/denis/samba_share/katrins_data/14809",
"/home/denis/samba_share/katrins_data/15174",
"/home/denis/samba_share/katrins_data/16742"]


Odd_dirs = ["/home/denis/samba_share/katrins_data/10147",
"/home/denis/samba_share/katrins_data/10967",
"/home/denis/samba_share/katrins_data/11432", # PET 3.0 
"/home/denis/samba_share/katrins_data/11749",
"/home/denis/samba_share/katrins_data/13544",
"/home/denis/samba_share/katrins_data/17775",
"/home/denis/samba_share/katrins_data/18103",
"/home/denis/samba_share/katrins_data/11240"] # careful here PET 2.145374449339207

# OK Dirs:
# Copy PET,CT and find the GTV and Relapse 


x = 0
ct_src = glob(OK_dirs[x] + '/' + "*ct_from*")
ct_dst = OK_dirs[x] + '/Cropped/CT.nii.gz'

pet_src = glob(OK_dirs[x] + '/' + "*pet_from*")
pet_dst = OK_dirs[x] + '/Cropped/PET.nii.gz'

gtv_src = glob(OK_dirs[x] + '/' + "*GTV.nii.gz")
gtv_dst = OK_dirs[x] + '/Cropped/GTV.nii.gz'

rel_src = glob(OK_dirs[x] + '/' + "*Relapse*")
rel_dst = OK_dirs[x] + '/Cropped/Relapse.nii.gz'


print("CT SOURCE: ", ct_src)
print("PET SOURCE: ", pet_src)
print("GTV SOURCE: ", gtv_src)
print("Relapse SOURCE: ",rel_src)

shutil.copyfile(ct_src, ct_dst)



### Notes
# Checkt if GTV exists 
# check the max()
# find all relapse 
# fuse relapse 
# binarize the relpase 

"""

for f in range (len(OneS_dirs)):
    #print(OneS_dirs_dirs[f])
    CT_dir = glob(OneS_dirs[f] + '/' +'*ct_from*')
    PET_dir = glob(OneS_dirs[f] + '/' +'*pet_from*')
    GTV_dir = glob(OneS_dirs[f] + '/' +'GTV.nii*')
    Relapse_dir = glob(OneS_dirs[f] + '/' +'Relapse*')


    CT = read_image(CT_dir)
    CT_cropped = CT[:,:, 1:]
    CT_itk = sitk.GetImageFromArray(CT_cropped)

    path = OneS_dirs + '/Cropped/'
    #sitk.WriteImage(CT_itk, path + 'CT.nii.gz')


"""
  

  

  




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




