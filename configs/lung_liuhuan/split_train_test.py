import os
import random

def split_train_test(data_root, train_ratio, trainlist_file, testlist_file):
    img_name = os.listdir(data_root)
    random.shuffle(img_name)
    nCount = 0
    trainlist_fp = open(trainlist_file, 'w')
    testlist_fp = open(testlist_file, 'w')
    while nCount < len(img_name) * train_ratio:
        trainlist_fp.write("{}\n".format(img_name[nCount].split('.')[0]))
        nCount += 1
    while nCount < len(img_name):
        testlist_fp.write("{}\n".format(img_name[nCount].split('.')[0]))
        nCount += 1
    trainlist_fp.close()
    testlist_fp.close()

if __name__ == '__main__':
    data_root = '/gruntdata/data/lung_cance_liuhuan/img'
    train_ratio = 0.8
    trainlist_file = '/gruntdata/data/lung_cance_liuhuan/trainlist.txt'
    testlist_file = '/gruntdata/data/lung_cance_liuhuan/testlist.txt'
    split_train_test(data_root, train_ratio, trainlist_file, testlist_file)
