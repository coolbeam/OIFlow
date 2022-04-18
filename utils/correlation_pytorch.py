import cv2
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.tools import tools, tensor_tools


class CorrTorch_lkm(tools.abstract_model):
    '''
    this is a correlation implemented by pytorch. The structure is the correlation proposed in PWC.
    The speed is 1/2 of the original cuda version. So this version is only used for debug.
    '''

    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1):
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.stride1 = stride1
        self.stride2 = stride2
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)

    def forward(self, in1, in2):
        bz, cn, hei, wid = in1.shape
        # print(self.kernel_size, self.pad_size, self.stride1)
        f1 = F.unfold(in1, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=self.stride1)
        f2 = F.unfold(in2, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=self.stride2)  # 在这一步抽取完了kernel以后做warping插值岂不美哉？
        # tensor_tools.check_tensor(in1, 'in1')
        # tensor_tools.check_tensor(in2, 'in2')
        # tensor_tools.check_tensor(f1, 'f1')
        # tensor_tools.check_tensor(f2, 'f2')
        matching_kernel_size = f2.shape[1]
        f2_ = torch.reshape(f2, (bz, matching_kernel_size, hei, wid))
        f2_ = torch.reshape(f2_, (bz * matching_kernel_size, hei, wid)).unsqueeze(1)
        # tensor_tools.check_tensor(f2_, 'f2_reshape')
        flag = False
        if flag:
            f2 = F.unfold(f2_, kernel_size=(hei, wid), padding=self.pad_size, stride=self.stride2)
            # tensor_tools.check_tensor(f2, 'f2_reunfold')
            _, kernel_number, window_number = f2.shape
            f2_ = torch.reshape(f2, (bz, matching_kernel_size, kernel_number, window_number))
            f2_2 = torch.transpose(f2_, dim0=1, dim1=3).transpose(2, 3)
        else:
            searching_kernel_size = self.max_hdisp * 2 + 1
            f2 = F.unfold(f2_, kernel_size=searching_kernel_size, padding=searching_kernel_size // 2, stride=self.stride2)
            # tensor_tools.check_tensor(f2, 'f2_reunfold')
            _, search_window_size, window_number = f2.shape
            f2_ = torch.reshape(f2, (bz, matching_kernel_size, search_window_size, window_number))
            f2_2 = torch.transpose(f2_, dim0=1, dim1=2)
            window_number = search_window_size

        f1_2 = f1.unsqueeze(1)
        # tensor_tools.check_tensor(f1_2, 'f1_2_reshape')
        # tensor_tools.check_tensor(f2_2, 'f2_2_reshape')
        res = f2_2 * f1_2
        res = torch.mean(res, dim=2)
        res = torch.reshape(res, (bz, window_number, hei, wid))
        # tensor_tools.check_tensor(res, 'res')
        return res

    @classmethod
    def demo(cls):
        im = np.random.random((1, 10, 100, 100))
        im_torch = torch.from_numpy(im).float()  # .cuda()
        CV = CorrTorch_lkm(kernel_size=1, max_displacement=4, stride1=1, stride2=1)
        res = CV(im_torch, im_torch)
        tensor_tools.check_tensor(im_torch, 'im_torch')
        tensor_tools.check_tensor(res, 'res')


if __name__ == '__main__':
    CorrTorch_lkm.demo()
