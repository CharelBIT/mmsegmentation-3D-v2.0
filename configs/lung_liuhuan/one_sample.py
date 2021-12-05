import os
import shutil
from tqdm import tqdm
def one_sample(data_root, label_root, dst_root, sample=100):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    img_id = os.listdir(data_root)[0].split('.')[0]
    print(img_id)
    for i in tqdm(range(sample)):
       shutil.copy(os.path.join(data_root, img_id + '.nii'),
                   os.path.join(dst_root, "{}_{}.nii".format(img_id, i)))
       print(os.path.join(label_root, "{}_lesion1.nii.gz".format(img_id)))
       shutil.copy(os.path.join(label_root, "{}_lesion1.nii.gz".format(img_id)),
                   os.path.join(dst_root, "{}_{}_lesion1.nii.gz".format(img_id, i)))

if __name__ == "__main__":
    data_root = '/gruntdata/data/lung_cance_liuhuan/img'
    label_root = '/gruntdata/data/lung_cance_liuhuan/roi_category'
    one_sample(data_root, label_root, "/gruntdata/data/lung_cance_liuhuan/one_sample")