"""
This module provides a utilitiy for reading and writing nii.gz images
"""

import SimpleITK as sitk

def volume_writer(img, file_path):
    """
    Writing MR volumes images in nii.gz format
    """
    img_gi = sitk.GetImageFromArray(img)
    sitk.WriteImage(img_gi, file_path)

def volume_reader(img_path):
    """
    Reading MR volume images in nii.gz format and
    converting to numpy arrays
    """
    img = sitk.ReadImage(img_path)
    return sitk.GetArrayFromImage(img)
