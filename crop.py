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



/home/denis/samba_share/katrins_data/10033/ct_from_-786_to_190.nii.gz
/home/denis/samba_share/katrins_data/10147/ct_from_-786_to_190.nii.gz
/home/denis/samba_share/katrins_data/10157/ct_from_-768_to_208.nii.gz
/home/denis/samba_share/katrins_data/10188/ct_from_-764_to_212.nii.gz
/home/denis/samba_share/katrins_data/10967/ct_from_-762_to_214.nii.gz
/home/denis/samba_share/katrins_data/11061/ct_from_-802_to_174.nii.gz
/home/denis/samba_share/katrins_data/11086/ct_from_-760.5_to_215.5.nii.gz
/home/denis/samba_share/katrins_data/11210/ct_from_-890_to_212.nii.gz
/home/denis/samba_share/katrins_data/11240/ct_from_-760_to_216.nii.gz
/home/denis/samba_share/katrins_data/11386/ct_from_-786_to_190.nii.gz
/home/denis/samba_share/katrins_data/11432/ct_from_-762_to_214.nii.gz
/home/denis/samba_share/katrins_data/11576/ct_from_-778_to_198.nii.gz
/home/denis/samba_share/katrins_data/11663/ct_from_-871_to_231.nii.gz
/home/denis/samba_share/katrins_data/11749/ct_from_-240_to_168.nii.gz
/home/denis/samba_share/katrins_data/13271/ct_from_-770_to_206.nii.gz
/home/denis/samba_share/katrins_data/13282/ct_from_-754_to_222.nii.gz
/home/denis/samba_share/katrins_data/13503/ct_from_-778_to_198.nii.gz
/home/denis/samba_share/katrins_data/13526/ct_from_-868_to_234.nii.gz
/home/denis/samba_share/katrins_data/13544/ct_from_-280_to_108.nii.gz
/home/denis/samba_share/katrins_data/13576/ct_from_-772_to_204.nii.gz
/home/denis/samba_share/katrins_data/13648/ct_from_-774_to_202.nii.gz
/home/denis/samba_share/katrins_data/13777/ct_from_-774_to_202.nii.gz
/home/denis/samba_share/katrins_data/14034/ct_from_-774_to_202.nii.gz
/home/denis/samba_share/katrins_data/14049/ct_from_-768_to_208.nii.gz
/home/denis/samba_share/katrins_data/14401/ct_from_-754_to_222.nii.gz
/home/denis/samba_share/katrins_data/14800/ct_from_-808_to_168.nii.gz
/home/denis/samba_share/katrins_data/14809/ct_from_-855.5_to_246.5.nii.gz
/home/denis/samba_share/katrins_data/15174/ct_from_-772_to_204.nii.gz
/home/denis/samba_share/katrins_data/16742/ct_from_-744_to_232.nii.gz
/home/denis/samba_share/katrins_data/17496/ct_from_-794_to_180.nii.gz
/home/denis/samba_share/katrins_data/17775/ct_from_-285.8_to_220.2.nii.gz
/home/denis/samba_share/katrins_data/18103/ct_from_-218_to_192.nii.gz
/home/denis/samba_share/katrins_data/6747/ct_from_-663_to_195.nii.gz
/home/denis/samba_share/katrins_data/6823/ct_from_-717_to_255.nii.gz
/home/denis/samba_share/katrins_data/7229/ct_from_-771_to_201.nii.gz
/home/denis/samba_share/katrins_data/7660/ct_from_-786_to_186.nii.gz
/home/denis/samba_share/katrins_data/8935/ct_from_-776_to_200.nii.gz
/home/denis/samba_share/katrins_data/9515/ct_from_-802_to_174.nii.gz
/home/denis/samba_share/katrins_data/9610/ct_from_-894_to_208.nii.gz
/home/denis/samba_share/katrins_data/9649/ct_from_-686_to_164.nii.gz
/home/denis/samba_share/katrins_data/9777/ct_from_-771_to_198.nii.gz
/home/denis/samba_share/katrins_data/9930/ct_from_-778_to_198.nii.gz
/home/denis/samba_share/katrins_data/9937/ct_from_-880_to_222.nii.gz


  

  

  




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




