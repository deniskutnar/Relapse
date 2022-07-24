import os
import SimpleITK as sitk
import numpy as np
from glob import glob
import shutil

import matplotlib.pyplot as plt
import skimage
from skimage.io import imsave
from skimage.transform import resize
from skimage.color import gray2rgb


# Step 1: Find CT, PET, GTV, Relapse with glob 
# Step 2: Print shape of all images in numpy
# Step 3: resample PET --> CT 
# Print figures to overlay CT and PET | GTV and Relapse 






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


ct_dirs = glob("/home/denis/samba_share/katrins_data/*/Processed/CT_CT*.nii.gz")
pet_dirs = glob("/home/denis/samba_share/katrins_data/*/Processed/PET_PET*.nii.gz")
gtv_dirs = glob("/home/denis/samba_share/katrins_data/*/Processed/GTV*.nii.gz")
relapse_dirs = glob("/home/denis/samba_share/katrins_data/*/Processed/Relapse*.nii.gz")

print(len(ct_dirs))
print(len(pet_dirs))
print(len(gtv_dirs))
print(len(relapse_dirs))

for i in range(len(ct_dirs)):
#for f in range(1):

    ct =  sitk.ReadImage(ct_dirs[i])
    pet = sitk.ReadImage(pet_dirs[i])
    pet = resize_image_itk(pet, ct, sitk.sitkLinear)
    gtv = sitk.ReadImage(gtv_dirs[i])
    relapse = sitk.ReadImage(relapse_dirs[i])

    ct = sitk.GetArrayFromImage(ct)
    pet = sitk.GetArrayFromImage(pet)
    gtv = sitk.GetArrayFromImage(gtv)
    relapse = sitk.GetArrayFromImage(relapse)



    title = ct_dirs[i][37:]

    f, ax = plt.subplots(2, 2, figsize=(10, 10))
    f.suptitle( str(title) , fontsize=16)
    ax[0][0].imshow(ct.max(0),  cmap = 'gray_r')
    ax[0][0].imshow(pet.max(0),  cmap = 'Reds', alpha=0.3)
    ax[0][1].imshow(ct.max(0), cmap = 'gray_r')
    ax[0][1].imshow(gtv.max(0), cmap = 'Reds', alpha=0.3)
    ax[0][1].imshow(relapse.max(0), cmap = 'Blues', alpha=0.3)

    ax[1][0].imshow(pet.max(0),  cmap = 'gray_r')
    ax[1][1].imshow(pet.max(0), cmap = 'gray_r')
    ax[1][1].imshow(gtv.max(0), cmap = 'Reds', alpha=0.4)
    ax[1][1].imshow(relapse.max(0), cmap = 'Blues', alpha=0.4)

    f.savefig("plots/"+ str(i) + ".png")
    plt.close()
    print(i)

exit()



ct_dir = "/home/denis/samba_share/katrins_data/6823/Cropped/CT.nii.gz"
pet_dir = "/home/denis/samba_share/katrins_data/6823/Cropped/PET.nii.gz"
mask_dir = "/home/denis/samba_share/katrins_data/6823/Cropped/GTV.nii.gz"


ct = read_image(ct_dir)
pet = read_image(pet_dir)
mask = read_image(mask_dir)


ct_slice = ct[ct.shape[0]//2, :, :]
#imsave('ct_slice.png', ct_slice)

pet_slice = pet[pet.shape[0]//2, :, :]
#imsave('pet_slice.png', pet_slice)

pet_slice_r = resize(pet_slice, ct_slice.shape)
#imsave('pet_slice_resized.png', pet_slice_r)

pet_ct_stack = np.hstack((pet_slice_r, ct_slice))
#imsave('pet_ct_stack.png', pet_ct_stack)

pet_high_suv = pet_slice_r > (np.mean(pet_slice_r) * 5)
imsave('pet_high_suv.png', pet_high_suv)
ct_with_high_suv = np.array(ct_slice)
ct_with_high_suv[pet_high_suv > 0] = np.max(ct_with_high_suv)
#imsave('ct_with_high_suv_10.png', ct_with_high_suv)


pet_ct_red_green = gray2rgb(ct_slice)
print('rgb image shape = ', pet_ct_red_green.shape)
print("CT max: ", ct_slice.max())
print("CT min: ", ct_slice.min())

print("PET max: ", pet_slice.max())
print("PET min: ", pet_slice.min())

ct_norm = ct_slice - np.min(ct_slice)
ct_norm = ct_norm / np.max(ct_norm)
pet_norm = pet_slice_r - np.min(pet_slice_r)
pet_norm = pet_norm / np.max(pet_norm)
pet_ct_red_green = gray2rgb(ct_norm)
pet_ct_red_green[:, :, 0] = ct_norm
pet_ct_red_green[:, :, 1] = pet_norm
pet_ct_red_green[:, :, 2] = ct_norm
imsave('pet_ct_red_green.png', pet_ct_red_green)



pet_ct_red_green[:, :, 0] = ct_norm
pet_ct_red_green[:, :, 1] = pet_norm
pet_ct_red_green[:, :, 2] = pet_norm > (np.mean(pet_norm) / 3)
imsave('6823_rgb.png', pet_ct_red_green)


exit()








