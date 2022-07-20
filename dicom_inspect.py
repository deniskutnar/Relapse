import sys
import os

import numpy as np
import pydicom
import nibabel as nib
from dicom_mask.convert import struct_to_mask


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

ct_dir = "/home/denis/samba_share/katrins_data/7229/CT"
pet_dir = "/home/denis/samba_share/katrins_data/7229/PET"

#ct = load_image_series(ct_dir)
pet = load_image_series(pet_dir)

print(pet)