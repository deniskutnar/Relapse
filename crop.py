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
"/home/denis/samba_share/katrins_data/7229",
"/home/denis/samba_share/katrins_data/7660",
"/home/denis/samba_share/katrins_data/9515",
"/home/denis/samba_share/katrins_data/9649",
"/home/denis/samba_share/katrins_data/9777",
"/home/denis/samba_share/katrins_data/9930",
"/home/denis/samba_share/katrins_data/11386",
"/home/denis/samba_share/katrins_data/17496"]

OneS_dirs = ["/home/denis/samba_share/katrins_data/8935", # missing relapse
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


x = 1

ct_src = glob(OneS_dirs[x] + '/' + "*ct_from*")
ct_src = ''.join(ct_src)
ct_dst = OneS_dirs[x] + '/Cropped/CT.nii.gz'
print(ct_src)

pet_src = glob(OneS_dirs[x] + '/' + "*pet_from*")
pet_src = ''.join(pet_src)
pet_dst = OneS_dirs[x] + '/Cropped/PET.nii.gz'

gtv_src = glob(OneS_dirs[x] + '/' + "*GTV.nii.gz")
gtv_src = ''.join(gtv_src)
gtv_dst = OneS_dirs[x] + '/Cropped/GTV.nii.gz'


rel_src1 = glob(OneS_dirs[x] + '/' + "*Relapse deformed_cau.nii*")
rel_src1 = ''.join(rel_src1)
rel_src2 = glob(OneS_dirs[x] + '/' + "*Relapse deformed_cran.nii*")
rel_src2 = ''.join(rel_src2)
rel_dst = OneS_dirs[x] + '/Cropped/Relapse.nii.gz'


rel_arr1 = read_image(rel_src1)
rel_arr2 = read_image(rel_src2)
rel_fuse = rel_arr1 + rel_arr2 

out_im = sitk.GetImageFromArray(rel_fuse)
Im = out_im
BinThreshImFilt = sitk.BinaryThresholdImageFilter()
BinThreshImFilt.SetLowerThreshold(1)
BinThreshImFilt.SetUpperThreshold(5)
BinThreshImFilt.SetOutsideValue(0)
BinThreshImFilt.SetInsideValue(1)
BinIm = BinThreshImFilt.Execute(Im)

CT = read_image(ct_src)
CT_cropped = CT[:,:, 1:]
CT_itk = sitk.GetImageFromArray(CT_cropped)

GTV = read_image(gtv_src)
GTV_cropped = GTV[:,:, 1:]
GTV_itk = sitk.GetImageFromArray(GTV_cropped)

Relp = sitk.GetArrayFromImage(BinIm)
Relp_cropped = Relp[:,:, 1:]
Relp_itk = sitk.GetImageFromArray(Relp_cropped)


#shutil.copy2(ct_src, ct_dst)
#shutil.copy2(pet_src, pet_dst)
#shutil.copy2(gtv_src, gtv_dst)
#shutil.copy2(rel_src, rel_dst)

sitk.WriteImage(CT_itk, ct_dst)
shutil.copy2(pet_src, pet_dst)
sitk.WriteImage(GTV_itk, gtv_dst)
sitk.WriteImage(Relp_itk, rel_dst)



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




