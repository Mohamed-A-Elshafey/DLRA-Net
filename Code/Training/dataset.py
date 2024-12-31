
import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata
import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata

from scipy.io import loadmat
import numpy as np




class HyperDataset(udata.Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
       
        if self.mode == 'train':
            self.h5f = h5py.File("data/train_ntire2022.h5", "r")
        elif self.mode == 'test':
            self.h5f = h5py.File("data/valid_ntire2022.h5", "r")
        if 'train' in self.mode:
            self.keys = list(self.h5f.keys())
            random.shuffle(self.keys)
            #self.len = 3500
            self.len = len(self.keys)
        else:
            self.keys = list(self.h5f.keys())
            self.keys.sort()
            self.len = len(self.keys)
            #self.len = 64
       
    def __len__(self):
        #return len(self.keys)
         return self.len

    def __getitem__(self, index):
        key = str(self.keys[index])
        data = np.array(self.h5f[key])
        data = torch.Tensor(data)
        return data[31:34,:,:], data[0:31,:,:]

    

    def close(self):
        self.h5f.close()

    def shuffle(self):
        if 'train' in self.mode:
            random.shuffle(self.keys)