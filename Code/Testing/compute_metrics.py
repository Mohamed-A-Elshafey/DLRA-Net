import glob
import os
import hdf5storage as hdf5
import numpy as np
import argparse
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio,structural_similarity



def compare_sam(x_true, x_pred):
    """
    :param x_true：(H, W, C)
    :param x_pred: (H, W, C)
    :return: sam_deg
    """
    num = 0
    sum_sam = 0
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
                sum_sam += np.arccos(
                    np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred)))
                num += 1
    sam_deg = (sum_sam / num) * 180 / np.pi
    return sam_deg


def compare_mpsnr(x_true, x_pred, data_range):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    channels = x_true.shape[2]
    total_psnr = [peak_signal_noise_ratio(image_true=x_true[:, :, k], image_test=x_pred[:, :, k], data_range=data_range)
                  for k in range(channels)]

    return np.mean(total_psnr)


def img_2d_mat(x_true, x_pred):
    """
    :param x_true: (H, W, C)
    :param x_pred: (H, W, C)
    :return: a matrix which shape is (C, H, W)
    """
    h, w, c = x_true.shape
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    x_mat = np.zeros((c, h * w), dtype=np.float32)
    y_mat = np.zeros((c, h * w), dtype=np.float32)
    for i in range(c):
        x_mat[i] = x_true[:, :, i].reshape((1, -1))
        y_mat[i] = x_pred[:, :, i].reshape((1, -1))
    return x_mat, y_mat


def compare_ergas(x_true, x_pred, ratio):
    """
    Calculate ERGAS, ERGAS offers a global indication of the quality of fused image.The ideal value is 0.
    :param x_true:
    :param x_pred:
    :param ratio: 
    :return:
    """
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    sum_ergas = 0
    for i in range(x_true.shape[0]):
        vec_x = x_true[i]
        vec_y = x_pred[i]
        err = vec_x - vec_y
        r_mse = np.mean(np.power(err, 2))
        tmp = r_mse / (np.mean(vec_x)**2)
        sum_ergas += tmp
    return (100 / ratio) * np.sqrt(sum_ergas / x_true.shape[0])

def compare_mssim(x_true, x_pred, data_range, multidimension):
    """
    :param x_true:
    :param x_pred:
    :param data_range:
    :param multidimension:
    :return:
    """
    mssim = [structural_similarity(im1=x_true[:, :, i], im2=x_pred[:, :, i], data_range=data_range, channel_axis=multidimension)
            for i in range(x_true.shape[2])]

    return np.mean(mssim)

def compute_MRAE(gt, rec):
    gt_hyper = gt
    rec_hyper = rec
    error = np.abs(rec_hyper - gt_hyper) / gt_hyper
    mrae = np.mean(error.reshape(-1))
    return mrae

def compute_RMSE(gt, rec):
    error = np.power(gt - rec, 2)
    rmse = np.sqrt(np.mean(error))
    return rmse

def main():
    path_rec = opt.path_rec
    path_gt = opt.path_gt

    name_rec_list = glob.glob(os.path.join(path_rec, '*.mat'))
    name_gt_list = glob.glob(os.path.join(path_gt, '*.mat'))
    name_rec_list.sort()
    name_gt_list.sort()

    mrae_all = []
    rmse_all = []
    sam_all = []
    mssim_all = []
    mpsnr_all = []
    ergas_all = []
    
    for i in range(len(name_gt_list)):
        
        hyper_rec = hdf5.loadmat(name_rec_list[i])['cube']
        hyper_gt = hdf5.loadmat(name_gt_list[i])['cube']
        if hyper_gt.min()<= 0.:
            print(os.path.basename(name_gt_list[i]), end=' ')
            print('This file is not suitable for compute the MRAE indicator.')
            continue
        hyper_rec = np.clip(hyper_rec, 0,1)
        mrae = compute_MRAE(hyper_gt, hyper_rec)
        rmse = compute_RMSE(hyper_gt, hyper_rec)
        sam = compare_sam(hyper_gt, hyper_rec)
        mpsnr = compare_mpsnr(x_true=hyper_gt, x_pred=hyper_rec, data_range=1)
        mssim = compare_mssim(x_true=hyper_gt, x_pred=hyper_rec, data_range=1,
                                     multidimension=False)
        ergas = compare_ergas(x_true=hyper_gt, x_pred=hyper_rec, ratio=4)
        

        print(os.path.basename(name_gt_list[i]), end=' ')
        print('mrae: '+str(mrae)+',  rmse: '+str(rmse)+',  sam: '+str(sam)+',  mssim: '+str(mssim) +',  mpsnr: '+str(mpsnr) +',  ergas: '+str(ergas))
        mrae_all.append(mrae)
        rmse_all.append(rmse)
        sam_all.append(sam)
        mpsnr_all.append(mpsnr)
        mssim_all.append(mssim)
        ergas_all.append(ergas)
    print('The average mrae is: '+str(sum(mrae_all)/len(mrae_all)))
    print('The average rmse is: '+str(sum(rmse_all)/len(rmse_all)))
    print('The average sam is: '+str(sum(sam_all)/len(sam_all)))
    print('The average mssim is: '+str(sum(mssim_all)/len(mssim_all)))
    print('The average mpsnr is: '+str(sum(mpsnr_all)/len(mpsnr_all)))
    print('The average ergas is: '+str(sum(ergas_all)/len(ergas_all)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SSR_test")
    parser.add_argument("--path_rec", type=str, default='./test_results', help="The path of the reconstructed valid spectral data.")
    parser.add_argument("--path_gt", type=str, default='./Valid_spectral', help="The path of the ground truth valid spectral data.")
    opt = parser.parse_args()
    main()