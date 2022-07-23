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


patient_no = 18103   # <----- Change me 


ct_dir = "/home/denis/samba_share/katrins_data/" + str(patient_no) + "/CT"
pet_dir = "/home/denis/samba_share/katrins_data/"+ str(patient_no) +"/PET"

#Create output folder 
folder_out = "/home/denis/samba_share/katrins_data/" + str(patient_no) + "/Processed/"
isExist = os.path.exists(folder_out)
if not isExist:
    os.makedirs(folder_out)
    print("The new directory is created!")


### Convert PET and CT 
## CT

dicom2nifti.convert_directory(ct_dir, folder_out)
ct_js_mv  = glob(folder_out + "*.nii.gz")
ct_js_mv = ''.join(ct_js_mv)
os.rename(ct_js_mv, folder_out + "CT_CT.nii.gz")
#ct = convert_dcm_2_nii_x(ct_dir, folder_out)

## PET
pet = convert_dcm_2_nii_x(pet_dir, folder_out)
pet_js_mv  = glob(folder_out + "*PET*.nii.gz")
pet_js_mv = ''.join(pet_js_mv)
os.rename(pet_js_mv, folder_out + "PET_PET.nii.gz")

### Remove Jason files 
pet_js = glob(folder_out + "*PET*.json")
pet_js_rm = ''.join(pet_js)
os.remove(pet_js_rm)

#ct_js  = glob(folder_out + "*CT*.json")
#ct_js_rm = ''.join(ct_js)
#os.remove(ct_js_rm)


### Path to CT and PET files
ct_nii_dir  = glob(folder_out + "*CT*.nii.gz") 
ct_nii_dir = ''.join(ct_nii_dir)
pet_nii_dir  = glob(folder_out + "*PET*.nii.gz")
pet_nii_dir = ''.join(pet_nii_dir)
ct  = sitk.ReadImage(ct_nii_dir)
pet  = sitk.ReadImage(pet_nii_dir)


### Get the GTV
gtv = get_struct_image(ct_dir, 'GTV_T')                  # <----- Change me 
#gtv = get_struct_image(ct_dir, 'GTV Radiolog')
gtv.CopyInformation(ct)
sitk.WriteImage(gtv, folder_out + 'GTV.nii.gz')

### Get the Relapses
"""
relapse = get_struct_image(ct_dir, 'Relapse Volume')  # <----- Change me 
relapse.CopyInformation(ct)
"""
relapse1 = get_struct_image(ct_dir, 'Relapse Volume_N')        # <----- Change me 
relapse1 = sitk.GetArrayFromImage(relapse1)
relapse2 = get_struct_image(ct_dir, 'Relapse Volume_T')        # <----- Change me 
relapse2 = sitk.GetArrayFromImage(relapse2)
#relapse3 = get_struct_image(ct_dir, 'Relapse volume_s')        # <----- Change me 
#relapse3 = sitk.GetArrayFromImage(relapse3)
rel_fuse = relapse1 + relapse2 

out_im = sitk.GetImageFromArray(rel_fuse)
Im = out_im
BinThreshImFilt = sitk.BinaryThresholdImageFilter()
BinThreshImFilt.SetLowerThreshold(1)
BinThreshImFilt.SetUpperThreshold(5)
BinThreshImFilt.SetOutsideValue(0)
BinThreshImFilt.SetInsideValue(1)
relapse = BinThreshImFilt.Execute(Im)
relapse.CopyInformation(ct)

sitk.WriteImage(relapse, folder_out + 'Relapse.nii.gz') 




### Get path to GTV and Relapse files 
gtv_nii_dir  = glob(folder_out + "*GTV.nii.gz")
gtv_nii_dir = ''.join(gtv_nii_dir)
relapse_nii_dir  = glob(folder_out + "*Relapse.nii.gz")    
relapse_nii_dir = ''.join(relapse_nii_dir)

### 4 Remove slices if needed 
# If one extra slices
"""
ct_crop = sitk.ReadImage(ct_nii_dir)[:, :, 1:]
gtv_crop = sitk.ReadImage(gtv_nii_dir)[:,:,1:]
#relapse_crop = sitk.ReadImage(relapse_nii_dir)[:,:,1:]
sitk.WriteImage(ct_crop, ct_nii_dir)
sitk.WriteImage(gtv_crop, folder_out + 'GTV.nii.gz')
#sitk.WriteImage(relapse_crop, folder_out + 'Relapse.nii.gz')

"""
# If odd
pet_crop = sitk.ReadImage(pet_nii_dir)[:,:,282:]
sitk.WriteImage(pet_crop, pet_nii_dir)




print(f'SIZE:')
print(f'CT: \t{ct.GetSize()} \nPET: \t{pet.GetSize()} \nGTV: \t{gtv.GetSize()} \nRelapse: \t{relapse.GetSize()}')
print('-' * 40)
print(f'SPACING:')
print(f'CT: \t{ct.GetSpacing()} \nPET: \t{pet.GetSpacing()} \nGTV: \t{gtv.GetSpacing()} \nRelapse: \t{relapse.GetSpacing()}') 
print('-' * 40)

"""
print(" After")
print(pet_crop.GetSize())
print(pet_crop.GetSpacing())
"""
print("after")

print(f'SIZE:')
print(f'CT: \t{ct_crop.GetSize()} \nPET: \t{pet.GetSize()} \nGTV: \t{gtv_crop.GetSize()} \nRelapse: \t{relapse_crop.GetSize()}')
print('-' * 50)
print(f'SPACING:')
print(f'CT: \t{ct_crop.GetSpacing()} \nPET: \t{pet.GetSpacing()} \nGTV: \t{gtv.GetSpacing()} \nRelapse: \t{relapse.GetSpacing()}') 
print('-' * 50)
































