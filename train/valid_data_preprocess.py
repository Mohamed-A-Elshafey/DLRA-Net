import os
import os.path
import cv2
import glob
import numpy as np
import argparse
import hdf5storage
import h5py

parser = argparse.ArgumentParser(description="SpectralSR")
parser.add_argument("--data_path", type=str, default='D:\\P.h.D\\Dataset', help="data path")
parser.add_argument("--out_data_path", type=str, default='./data', help="out data path")


opt = parser.parse_args()


def main():
    if not os.path.exists(opt.out_data_path):
        os.makedirs(opt.out_data_path)
    h5f = h5py.File('./data/valid_ntire2022.h5', 'w')


    process_data(h5f,mode='valid')


def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)


def process_data(h5f,mode):
    if mode == 'valid':
        print("\nprocess validation set ...\n")
        patch_num = 1
        filenames_hyper = glob.glob(os.path.join(opt.data_path, 'Valid_spectral', '*.mat'))
        filenames_rgb = glob.glob(os.path.join(opt.data_path, 'Valid_RGB', '*.jpg'))
        filenames_hyper.sort()
        filenames_rgb.sort()
        # for k in range(1):  # make small dataset
        for k in range(len(filenames_rgb)):
            print([filenames_rgb[k]])
            # load hyperspectral image
            mat = hdf5storage.loadmat(filenames_hyper[k])
            hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [2, 0, 1])
            if hyper.min() <= 0:
                print('This file contains non-positive values and is not suitable for Testing!')
                continue
            hyper = normalize(hyper, max_val=1., min_val=0.)
            # load rgb image
            rgb = cv2.imread(filenames_rgb[k])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = np.transpose(rgb, [2, 0, 1])
            rgb = normalize(np.float32(rgb), max_val=rgb.max(), min_val=0.)
            print("generate valid sample #%d" % patch_num)
            data = np.concatenate((hyper[:,:-2,:], rgb[:,:-2,:]), 0)
            h5f.create_dataset(str(patch_num), data=data)
            patch_num += 1

        print("\Validation set: # samples %d\n" % (patch_num-1))


if __name__ == '__main__':
    main()

