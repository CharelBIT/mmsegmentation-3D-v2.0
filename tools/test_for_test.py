import nibabel as nib
from nibabel.processing import resample_to_output
img_nii = nib.load("/gruntdata/data/BraTS17/MICCAI_BraTS17_Data_Training/HGG/Brats17_CBICA_AQO_1/"
                   "Brats17_CBICA_AQO_1_flair.nii.gz")

print(img_nii.shape)

img_nii = resample_to_output(img_nii, voxel_sizes=(2., 2., 2.))
print(img_nii.shape)
print(img_nii.header)
print(img_nii.affine)
import torch
torch.optim.adam