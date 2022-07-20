import os
import SimpleITK as sitk
import numpy as np
from glob import glob
import shutil
from pydicom import dcmread
import matplotlib.pyplot as plt
import skimage
from skimage.io import imsave






def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkNearestNeighbor):
    """
    use itk Method to convert the original image resample To be consistent with the target image
    :param ori_img: Original alignment required itk image
    :param target_img: Target to align itk image
    :param resamplemethod: itk interpolation method : sitk.sitkLinear-linear  sitk.sitkNearestNeighbor-Nearest neighbor
    :return:img_res_itk: Resampling okay itk image
    """
    target_Size = target_img.GetSize()      # Target image size [x,y,z]
    target_Spacing = target_img.GetSpacing()   # Voxel block size of the target [x,y,z]
    target_origin = target_img.GetOrigin()      # Starting point of target [x,y,z]
    target_direction = target_img.GetDirection()  # Target direction [crown, sagittal, transverse] = [z,y,x]

    # The method of itk is resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # Target image to resample
    # Set the information of the target image
    resampler.SetSize(target_Size)		# Target image size
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # Set different dype according to the need to resample the image
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt16)   # Nearest neighbor interpolation is used for mask, and uint16 is saved
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # Linear interpolation is used for PET/CT/MRI and the like, and float32 is saved
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # Get the resampled image
    return itk_img_resampled

def read_image(path):
    img = sitk.ReadImage(path)
    img_as_numpy = sitk.GetArrayFromImage(img).astype('float32')
    return img_as_numpy



ct_dir = "/home/denis/samba_share/katrins_data/7229/Cropped/CT.nii.gz"
pet_dir = "/home/denis/samba_share/katrins_data/7229/Cropped/PET.nii.gz"
mask_dir = "/home/denis/samba_share/katrins_data/7229/Cropped/GTV.nii.gz"




## resample PET --> CT
#t_img = sitk.ReadImage(ct_dir)
#o_img = sitk.ReadImage(pet_dir)
#reg_pet = resize_image_itk(o_img, t_img, sitk.sitkLinear)

ct = read_image(ct_dir)
pet = read_image(pet_dir)
mask = read_image(mask_dir)

print(ct.shape)
print(pet.shape)
print(mask.shape)

ct_slice = ct[ct.shape[0]//2, :, :]
imsave('ct_slice.png', ct_slice)

exit()

f, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0][0].imshow(ct.max(0),  cmap = 'gray')
ax[0][0].imshow(pet.max(0),  cmap = 'Reds', alpha=0.3)
ax[0][1].imshow(ct.max(0), cmap = 'gray')
ax[0][1].imshow(mask.max(0), cmap = 'Reds', alpha=0.3)

ax[1][0].imshow(pet.max(0),  cmap = 'gray')
ax[1][1].imshow(pet.max(0), cmap = 'gray')
ax[1][1].imshow(mask.max(0), cmap = 'Reds', alpha=0.3)

f.savefig("plot.png")




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

"/home/denis/samba_share/katrins_data/14809", # missing relapse 
"/home/denis/samba_share/katrins_data/15174", # missing relapse 
"/home/denis/samba_share/katrins_data/16742"]


Odd_dirs = ["/home/denis/samba_share/katrins_data/10147", # CT 2.0 PET 2.712871287128713
"/home/denis/samba_share/katrins_data/10967", # PET 1.4 
"/home/denis/samba_share/katrins_data/11432", # PET 3.0 
"/home/denis/samba_share/katrins_data/11749",
"/home/denis/samba_share/katrins_data/13544",
"/home/denis/samba_share/katrins_data/17775",
"/home/denis/samba_share/katrins_data/18103",
"/home/denis/samba_share/katrins_data/11240"] # careful here PET 2.145374449339207

# OK Dirs:
# Copy PET,CT and find the GTV and Relapse 


x = 0

ct_src = glob(Odd_dirs[x] + '/' + "*ct_from*")
ct_src = ''.join(ct_src)

"""

ct_src = glob(Odd_dirs[x] + '/' + "*ct_from*")
ct_src = ''.join(ct_src)
ct_dst = Odd_dirs[x] + '/Cropped/CT.nii.gz'
print(ct_src)

pet_src = glob(Odd_dirs[x] + '/' + "*pet_from*")
pet_src = ''.join(pet_src)
pet_dst = Odd_dirs[x] + '/Cropped/PET.nii.gz'

gtv_src = glob(Odd_dirs[x] + '/' + "*GTV.nii.gz")
gtv_src = ''.join(gtv_src)
gtv_dst = Odd_dirs[x] + '/Cropped/GTV.nii.gz'


rel_src1 = glob(Odd_dirs[x] + '/' + "*Relapse*")
rel_src1 = ''.join(rel_src1)
rel_dst = Odd_dirs[x] + '/Cropped/Relapse.nii.gz'


rel_src2 = glob(Odd_dirs[x] + '/' + "*relapse volume_R.nii*")
rel_src2 = ''.join(rel_src2)

#rel_src3 = glob(Odd_dirs[x] + '/' + "*Relapse volume_s.nii*")
#rel_src3 = ''.join(rel_src3)



rel_arr1 = read_image(rel_src1)
rel_arr2 = read_image(rel_src2)
#rel_arr3 = read_image(rel_src3)
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
CT_cropped = CT[:,:, 333:435]
print("CT cropped ", CT_cropped.shape)
CT_itk = sitk.GetImageFromArray(CT_cropped)

GTV = read_image(gtv_src)
GTV_cropped = GTV[:,:, 333:435]
GTV_itk = sitk.GetImageFromArray(GTV_cropped)

Relp = read_image(rel_src1)
Relp_cropped = Relp[:,:, 333:435]
Relp_itk = sitk.GetImageFromArray(Relp_cropped)

#Relp = sitk.GetArrayFromImage(BinIm)
#Relp_cropped = Relp[:,:, 1:]
#Relp_itk = sitk.GetImageFromArray(Relp_cropped)


#shutil.copy2(ct_src, ct_dst)
#shutil.copy2(pet_src, pet_dst)
#shutil.copy2(gtv_src, gtv_dst)
#shutil.copy2(rel_src, rel_dst)

sitk.WriteImage(CT_itk, ct_dst)
shutil.copy2(pet_src, pet_dst)
sitk.WriteImage(GTV_itk, gtv_dst)
sitk.WriteImage(Relp_itk, rel_dst)
"""


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




