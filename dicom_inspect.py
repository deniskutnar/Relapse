import sys
import os

import numpy as np
import pydicom
import nibabel as nib
from dicom_mask.convert import struct_to_mask
import SimpleITK as sitk
import dicom2nifti


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


def load_image_series(dicom_dir):
    """
    Get all dicom image dataset files for a dicom series in a dicom dir.
    """
    image_series = []
    dicom_files = os.listdir(dicom_dir)
    for f in dicom_files:
        fpath = os.path.join(dicom_dir, f)
        if os.path.isfile(fpath):
            fdataset = pydicom.dcmread(fpath, force=True)   ## Read slice
            # Computed Radiography Image Storage SOP Class UID
            # https://dicom.nema.org/dicom/2013/output/chtml/part04/sect_B.5.html
            mr_sop_class_uid = '1.2.840.10008.5.1.4.1.1.4'
            ct_sop_class_uid = '1.2.840.10008.5.1.4.1.1.2'
            pet = '1.2.840.10008.5.1.4.1.1.128'
            enhanced_pet = '1.2.840.10008.5.1.4.1.1.130'
            legacy_pet = '1.2.840.10008.5.1.4.1.1.128.1'

            #print('transfer syntax = ', fdataset.file_meta.TransferSyntaxUID)
            if not hasattr(fdataset.file_meta, 'TransferSyntaxUID'):
                # fall back to default TransferSyntaxUID
                # https://dicomlibrary.com/dicom/transfer-syntax/
                # 1.2.840.10008.1.2	Implicit VR Endian: Default Transfer Syntax for DICOM
                fdataset.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2' 

            if fdataset.SOPClassUID in [mr_sop_class_uid, ct_sop_class_uid,
                                        pet, enhanced_pet, legacy_pet]:
                image_series.append(fdataset)
    image_series = sorted(image_series, key=lambda s: s.SliceLocation) # slice location 
    return image_series



def get_scan_image(dicom_series_path):
    """ return dicom images as 3D numpy array 
        warning: this function assumes the file names
                 sorted alpha-numerically correspond to the
                 position of the dicom slice in the z-dimension (depth)
                 if this assumption does not hold then you may need
                 to sort them based on their metadata (actual position in space).
    """
    image_series_files = load_image_series(dicom_series_path)
    first_im = image_series_files[0]
    height, width = first_im.pixel_array.shape
    depth = len(image_series_files)
    image = np.zeros((depth, height, width))
    for i, im in enumerate(image_series_files):
        image[i] = im.pixel_array ## pixel - SUV value , as array 
    return image

def get_struct_image(dicom_series_path, struct_name):
    dicom_files = [d for d in os.listdir(dicom_series_path) if d.endswith('.dcm')]

    # We assume here that you identified a single struct for each fraction
    # and given it the same name in all fractions in order for it to be exported.
    # This may require a pre-processing or manual checking to ensure that
    # your structs of interest all have the same names.
    mask = struct_to_mask(dicom_series_path, dicom_files, struct_name)
    mask = np.flip(mask, axis=0)
    if not np.any(mask):
        raise Exception(f'Struct with name {struct_name} was not found in {dicom_series_path}'
                        ' or did not contain any delineation data.'
                        ' Are you sure that all structs of interest are named '
                        'consistently and non-empty?')
    return mask



ct_dir = "/home/denis/samba_share/katrins_data/7229/CT"
pet_dir = "/home/denis/samba_share/katrins_data/7229/PET"



#dicom2nifti.convert_directory(ct_dir, 'CT/')

ct  = sitk.ReadImage('CT/5_ct__30mm_b40f.nii.gz')
ct_numpy = sitk.GetArrayFromImage(ct)
print(ct_numpy.shape)



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




