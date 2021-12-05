import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd

def struct_label(label_csv, save_csv):
    org_label_df = pd.read_csv(label_csv)
    dst_label_df = pd.DataFrame()
    for i in range(org_label_df.shape[0]):
        if not pd.isna(org_label_df.iloc[i]["Label_0"]):
            img_id = org_label_df.iloc[i]["Label_0"].split('\\')[-1].split('_')[0]
            dst_label_df.loc[img_id, 'Label'] = 0
        if not pd.isna(org_label_df.iloc[i]["Label_1"]):
            img_id = org_label_df.iloc[i]["Label_1"].split('\\')[-1].split('_')[0]
            dst_label_df.loc[img_id, 'Label'] = 1
    dst_label_df.to_csv(save_csv)


def category_label(label_csv, label_root, dst_root):
    label_df = pd.read_csv(label_csv, dtype=str)
    # print(label_df.head())
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    for i in tqdm(range(label_df.shape[0])):
        ser = label_df.iloc[i]
        img_id = ser[0]
        label = float(ser[1])
        img_nii = nib.load(os.path.join(label_root, '{}_lesion1.nii.gz'.format(img_id)))
        img = np.squeeze(img_nii.get_fdata(dtype=np.float32))
        dst_img_name = os.path.join(dst_root, '{}_lesion1.nii.gz'.format(img_id))
        print("[DEBUG] max: {}, min: {}".format(img.max(), img.min()))
        img[img > 0] = label + 1
        dst_img_nii = nib.Nifti1Image(img, img_nii.affine)
        nib.save(dst_img_nii, dst_img_name)

if __name__ == '__main__':
    # label_csv = '/gruntdata/data/lung_cance_liuhuan/label.csv'
    # save_csv = '/gruntdata/data/lung_cance_liuhuan/id2label.csv'
    # struct_label(label_csv, save_csv)
    label_csv = '/gruntdata/data/lung_cance_liuhuan/id2label.csv'
    label_root = '/gruntdata/data/lung_cance_liuhuan/roi'
    dst_root = '/gruntdata/data/lung_cance_liuhuan/roi_category'
    category_label(label_csv, label_root, dst_root)

