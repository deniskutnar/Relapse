import sys
import os

import numpy as np
import pydicom
import nibabel as nib
from dicom_mask.convert import struct_to_mask
import SimpleITK as sitk
import dicom2nifti

from nipype.interfaces.dcm2nii import Dcm2niix
from glob import glob

import skimage
from skimage.io import imsave
from skimage.transform import resize
from skimage.color import gray2rgb

import matplotlib.pyplot as plt


# Step 1: Iterate through the slices in the PET image and print the slice location and max suv value.
# Step 2: take note of the slice location of a slice with a high suv value (max if obvious)
# Step 3: Iterate through the CT image slices and see if a slice is available with the corresponding slice location to the PET slice.
# Step 4: Show the CT slice and PET slice in a combined way (using code that shows them in RGB)
#         This should have the same problem as in the slices may not align exactly.
# Step 5: Inspect the properties of the dicom fdataset object to get the spacing information
#          including spacing information in x and y.
#          Also try to find the information that tells us about the slice position relative to patient origin in x and y.
#          The idea is to inspect the dicom data and read the dicom documentation to try to figure out how
#          the data should be cropped and scaled to make it consistent.




# 1. convert CT with 'Dcm2niix'
# 2. get structures with 'get_struct_image' and copy info
# 3. convert PET with 'Dcm2niix'
# 4. remove extra slices from CT 
# 5. Resample PET ---> CT with 'resize_image_itk'
# 6. Plot RGB overlay 
# 7. Normalize + Clip the data (HU)
# 7. Create hdf5




def convert_dcm_2_nii_x(dcm_folder, output_folder):    
    converter = Dcm2niix()    
    converter.inputs.source_dir = dcm_folder    
    converter.inputs.output_dir = output_folder    
    converter.inputs.compress = 'i'    
    converter.run()

def get_struct_image(dicom_series_path, struct_name):
    dicom_files = [d for d in os.listdir(dicom_series_path) if d.endswith('.dcm')]

    # We assume here that you identified a single struct for each fraction
    # and given it the same name in all fractions in order for it to be exported.
    # This may require a pre-processing or manual checking to ensure that
    # your structs of interest all have the same names.
    mask = struct_to_mask(dicom_series_path, dicom_files, struct_name)
    mask = np.flip(mask, axis=1)
    if not np.any(mask):
        raise Exception(f'Struct with name {struct_name} was not found in {dicom_series_path}'
                        ' or did not contain any delineation data.'
                        ' Are you sure that all structs of interest are named '
                        'consistently and non-empty?')
    mask_itk = sitk.GetImageFromArray(mask)

    Im = mask_itk
    BinThreshImFilt = sitk.BinaryThresholdImageFilter()
    BinThreshImFilt.SetLowerThreshold(1)
    BinThreshImFilt.SetUpperThreshold(5)
    BinThreshImFilt.SetOutsideValue(0)
    BinThreshImFilt.SetInsideValue(1)
    BinMask = BinThreshImFilt.Execute(Im)

    return BinMask

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


patient_no = 7229      # <----- Change me 

ct_dir = "/home/denis/samba_share/katrins_data/" + str(patient_no) + "/CT"
pet_dir = "/home/denis/samba_share/katrins_data/"+ str(patient_no) +"/PET"

#Create output folder 
folder_out = "/home/denis/samba_share/katrins_data/" + str(patient_no) + "/Processed/"
isExist = os.path.exists(folder_out)
if not isExist:
    os.makedirs(folder_out)
    print("The new directory is created!")


### Concert PET and CT 
pet = convert_dcm_2_nii_x(pet_dir, folder_out)
ct = convert_dcm_2_nii_x(ct_dir, folder_out)
### Remove Jason files 
ct_js  = glob(folder_out + "*CT*.json")
ct_js_rm = ''.join(ct_js)
os.remove(ct_js_rm)
pet_js = glob(folder_out + "*PET*.json")
pet_js_rm = ''.join(pet_js)
os.remove(pet_js_rm)


### Path to CT and PET files
ct_nii_dir  = glob(folder_out + "*CT*.nii.gz")
ct_nii_dir = ''.join(ct_nii_dir)
pet_nii_dir  = glob(folder_out + "*PET*.nii.gz")
pet_nii_dir = ''.join(pet_nii_dir)
ct  = sitk.ReadImage(ct_nii_dir)

### Get the GTV
#gtv = get_struct_image(ct_dir, 'CTV T')                  # <----- Change me 
gtv = get_struct_image(ct_dir, 'GTV Radiolog')
gtv.CopyInformation(ct)
sitk.WriteImage(gtv, folder_out + 'GTV.nii.gz')

### Get the Relapses
relapse = get_struct_image(ct_dir, 'Relapse deformed')  # <----- Change me 
relapse.CopyInformation(ct)
#relapse = get_struct_image(ct_dir, 'Relapse 1')        # <----- Change me 
#relapse.CopyInformation(ct)
sitk.WriteImage(relapse, folder_out + 'Relapse.nii.gz')

### Get path to GTV and Relapse files 
gtv_nii_dir  = glob(folder_out + "*GTV.nii.gz")
gtv_nii_dir = ''.join(gtv_nii_dir)
relapse_nii_dir  = glob(folder_out + "*Relapse.nii.gz")
relapse_nii_dir = ''.join(relapse_nii_dir)

### 4 Remove slices if needed 
ct  = read_image(ct_nii_dir)
pet = read_image(pet_nii_dir)
gtv = read_image(gtv_nii_dir)
relapse = read_image(relapse_nii_dir)
print("CT:      ", ct.shape)
print("PET:     ", pet.shape)
print("GTV:     ", gtv.shape)
print("Relapse: ", relapse.shape)

# if z axes is equal do nothing
if len(ct) == len(pet):
    print("No cropping!!!")
# if one extra slices
#ct = ct[1:,:,:]
#gtv = gtv[1:,:,:]
#relapse = relapse[1:,:,:]

### overrite 



exit()
pet = sitk.ReadImage(pet_nii_dir)
pet = resize_image_itk(pet, ct, sitk.sitkLinear)
mask = sitk.ReadImage(gtv_nii_dir)
ct = sitk.GetArrayFromImage(ct)
pet = sitk.GetArrayFromImage(pet)
mask = sitk.GetArrayFromImage(mask)

f, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0][0].imshow(ct.max(0),  cmap = 'gray')
ax[0][0].imshow(pet.max(0),  cmap = 'Reds', alpha=0.3)
ax[0][1].imshow(ct.max(0), cmap = 'gray')
ax[0][1].imshow(mask.max(0), cmap = 'Reds', alpha=0.3)
ax[1][0].imshow(pet.max(0),  cmap = 'gray')
ax[1][1].imshow(pet.max(0), cmap = 'gray')
ax[1][1].imshow(mask.max(0), cmap = 'Reds', alpha=0.3)
f.savefig("XXX_res.png")

exit()







print(f'SIZE:')
print(f'CT: \t{ct.GetSize()} \nPET: \t{pet.GetSize()} \nGTV: \t{gtv.GetSize()} \nRelapse: \t{relapse.GetSize()}')
print('-' * 40)
print(f'SPACING:')
print(f'CT: \t{ct.GetSpacing()} \nPET: \t{pet.GetSpacing()} \nGTV: \t{gtv.GetSpacing()} \nRelapse: \t{relapse.GetSpacing()}') 
print('-' * 40)
print(f'ORIGIN:')
print(f'CT: \t{ct.GetOrigin()} \nPET: \t{pet.GetOrigin()} \nGTV: \t{gtv.GetOrigin()} \nRelapse: \t{relapse.GetOrigin()}') 
print('-' * 40)
print(f'DIRECTION:')
print(f'CT: \t{ct.GetDirection()} \nPET: \t{pet.GetDirection()} \nGTV: \t{gtv.GetDirection()} \nRelapse: \t{relapse.GetDirection()}') 

print("")

### Resample 
pet_res = resize_image_itk(pet, ct, sitk.sitkLinear)


print(f'SIZE:')
print(f'CT: \t{ct.GetSize()} \nPET: \t{pet_res.GetSize()} \nGTV: \t{gtv.GetSize()} \nRelapse: \t{relapse.GetSize()}')
print('-' * 40)
print(f'SPACING:')
print(f'CT: \t{ct.GetSpacing()} \nPET: \t{pet_res.GetSpacing()} \nGTV: \t{gtv.GetSpacing()} \nRelapse: \t{relapse.GetSpacing()}') 
print('-' * 40)
print(f'ORIGIN:')
print(f'CT: \t{ct.GetOrigin()} \nPET: \t{pet_res.GetOrigin()} \nGTV: \t{gtv.GetOrigin()} \nRelapse: \t{relapse.GetOrigin()}') 
print('-' * 40)
print(f'DIRECTION:')
print(f'CT: \t{ct.GetDirection()} \nPET: \t{pet_res.GetDirection()} \nGTV: \t{gtv.GetDirection()} \nRelapse: \t{relapse.GetDirection()}') 











exit()


print(f'SIZE:')
print(f'CT: \t{ct.GetSize()}')
print('-' * 40)
print(f'SPACING:')
print(f'CT: \t{ct.GetSpacing()}') 
print('-' * 40)
print(f'ORIGIN:')
print(f'CT: \t{ct.GetOrigin()}') 
print('-' * 40)
print(f'DIRECTION:')
print(f'CT: \t{ct.GetDirection()}') 

print("")

print(f'SIZE:')
print(f'CT: \t{pet.GetSize()}')
print('-' * 40)
print(f'SPACING:')
print(f'CT: \t{pet.GetSpacing()}') 
print('-' * 40)
print(f'ORIGIN:')
print(f'CT: \t{pet.GetOrigin()}') 
print('-' * 40)
print(f'DIRECTION:')
print(f'CT: \t{pet.GetDirection()}') 





exit()

pet  = sitk.ReadImage('PET/PET_PET_3mm_AC_20090721101551_6.nii.gz')
print(f'SIZE:')
print(f'CT: \t{pet.GetSize()}')
print('-' * 40)
print(f'SPACING:')
print(f'CT: \t{pet.GetSpacing()}') 
print('-' * 40)
print(f'ORIGIN:')
print(f'CT: \t{pet.GetOrigin()}') 
print('-' * 40)
print(f'DIRECTION:')
print(f'CT: \t{pet.GetDirection()}') 



exit()

#dicom2nifti.convert_directory(ct_dir, 'CT/')
dicom2nifti.convert_directory(pet_dir, 'PET/pet.nii.gz')


exit ()

ct  = sitk.ReadImage('CT/5_ct__30mm_b40f.nii.gz')
ct_numpy = sitk.GetArrayFromImage(ct)
print(ct_numpy.shape)

mask = get_struct_image(ct_dir, 'GTV Radiolog')
mask = sitk.GetImageFromArray(mask)
mask.CopyInformation(ct)


print(f'SIZE:')
print(f'CT: \t{ct.GetSize()}')
print('-' * 40)
print(f'SPACING:')
print(f'CT: \t{ct.GetSpacing()}') 
print('-' * 40)
print(f'ORIGIN:')
print(f'CT: \t{ct.GetOrigin()}') 
print('-' * 40)
print(f'DIRECTION:')
print(f'CT: \t{ct.GetDirection()}') 

print(f'SIZE:')
print(f'CT: \t{mask.GetSize()}')
print('-' * 40)
print(f'SPACING:')
print(f'CT: \t{mask.GetSpacing()}') 
print('-' * 40)
print(f'ORIGIN:')
print(f'CT: \t{mask.GetOrigin()}') 
print('-' * 40)
print(f'DIRECTION:')
print(f'CT: \t{mask.GetDirection()}') 



exit()

#ct = load_image_series(ct_dir)
#pet = load_image_series(pet_dir)

ct_image = get_scan_image(ct_dir)
ct_itk = sitk.GetImageFromArray(ct_image)

mask = get_struct_image(ct_dir, 'GTV Radiolog')
mask = sitk.GetImageFromArray(mask)

print(mask)

print(f'SIZE:')
print(f'CT: \t{mask.GetSize()}')
print('-' * 40)
print(f'SPACING:')
print(f'CT: \t{mask.GetSpacing()}') 
print('-' * 40)
print(f'ORIGIN:')
print(f'CT: \t{mask.GetOrigin()}') 
print('-' * 40)
print(f'DIRECTION:')
print(f'CT: \t{mask.GetDirection()}') 



exit()

print(f'SIZE:')
print(f'CT: \t{ct_itk.GetSize()}')
print('-' * 40)
print(f'SPACING:')
print(f'CT: \t{ct_itk.GetSpacing()}') 
print('-' * 40)
print(f'ORIGIN:')
print(f'CT: \t{ct_itk.GetOrigin()}') 
print('-' * 40)
print(f'DIRECTION:')
print(f'CT: \t{ct_itk.GetDirection()}') 




