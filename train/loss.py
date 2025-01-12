#!/usr/local/bin/python

from __future__ import division
import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, labels):
        error = torch.abs(outputs - labels) / labels
        rrmse = torch.mean(error.view(-1))
        struct_tensor_loss = self.count_struct_tensor_v1(outputs.data, labels.data)
        return rrmse, struct_tensor_loss

    def count_struct_tensor_v1(self, outputs, labels):
        b, c, h, w, = outputs.shape
        outputs = outputs.view(b * c, h, w).unsqueeze(0).unsqueeze(0)
        labels = labels.view(b * c, h, w).unsqueeze(0).unsqueeze(0)
        gx_kernel = torch.Tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).cuda()
        gy_kernel = torch.Tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).cuda()
        gradx = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                          bias=False).cuda()
        gradx.weight.data = gx_kernel.view(1, 1, 1, 3, 3)
        grady = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                          bias=False).cuda()
        grady.weight.data = gy_kernel.view(1, 1, 1, 3, 3)

        with torch.no_grad():
            imx = gradx(outputs)
            imy = grady(outputs)
        M00, M01, M11 = imx * imx, imx * imy, imy * imy
        outputs_e1 = (M00 + M11) / 2 + torch.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2

        with torch.no_grad():
            imx = gradx(labels)
            imy = grady(labels)
        M00, M01, M11 = imx * imx, imx * imy, imy * imy
        labels_e1 = (M00 + M11) / 2 + torch.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2

        loss_fn = nn.L1Loss()
        errorr = loss_fn(outputs_e1, labels_e1).sqrt()
        return errorr

