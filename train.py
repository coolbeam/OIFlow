import sys
import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from network.pwc_irr import IRRPWC, Flow_test_model
# from datasets.flow_dataset import FlowDataset
from datasets.evaluation import Training_eval_manager
from utils.tools import tools, tensor_tools, file_tools
from datasets.unsup_dataset import UnFlowDataset


class Unsup_Loss(tools.abstract_config):
    class Occ_Check_Model(tools.abstract_config):

        def __init__(self, **kwargs):
            self.occ_type = 'for_back_check'
            self.occ_alpha_1 = 1.0
            self.occ_alpha_2 = 0.05
            self.obj_out_all = 'obj'  # obj, out, all, when boundary dilated warping is used, this should be 'obj'

            self.update(kwargs)
            self._check()

        def _check(self):
            assert self.occ_type in ['for_back_check', 'forward_warp']
            assert self.obj_out_all in ['obj', 'out', 'all']

        def __call__(self, flow_f, flow_b):
            '''
            input is optical flow. Using forward-backward checking to compute occlusion regions. 0 stands for occ region. 1 is for other regions.
            '''
            # regions that moving out of the image plane
            if self.obj_out_all == 'out':
                out_occ_fw = self.torch_outgoing_occ_check(flow_f)
                out_occ_bw = self.torch_outgoing_occ_check(flow_b)
                return out_occ_fw, out_occ_bw

            # all occlusion regions
            if self.occ_type == 'for_back_check':
                occ_1, occ_2 = self._forward_backward_occ_check(flow_fw=flow_f, flow_bw=flow_b)
            elif self.occ_type == 'forward_warp':
                raise ValueError('not implemented')
            else:
                raise ValueError('not implemented occlusion check method: %s' % self.occ_type)

            if self.obj_out_all == 'all':
                return occ_1, occ_2

            # 'out' regions are not considered as occlusion
            if self.obj_out_all == 'obj':
                out_occ_fw = self.torch_outgoing_occ_check(flow_f)
                out_occ_bw = self.torch_outgoing_occ_check(flow_b)
                obj_occ_fw = self.torch_get_obj_occ_check(occ_mask=occ_1, out_occ=out_occ_fw)
                obj_occ_bw = self.torch_get_obj_occ_check(occ_mask=occ_2, out_occ=out_occ_bw)
                return obj_occ_fw, obj_occ_bw

            raise ValueError("obj_out_all should be in ['obj','out','all'], but get: %s" % self.obj_out_all)

        def _forward_backward_occ_check(self, flow_fw, flow_bw):
            """
            In this function, the parameter alpha needs to be checked
            # 0 means the occlusion region where the photo loss we should ignore
            """

            def length_sq(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.pow(temp, 0.5)
                # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                return temp

            mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
            flow_bw_warped = tensor_tools.torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
            flow_fw_warped = tensor_tools.torch_warp(flow_fw, flow_bw)
            flow_diff_fw = flow_fw + flow_bw_warped
            flow_diff_bw = flow_bw + flow_fw_warped
            occ_thresh = self.occ_alpha_1 * mag_sq + self.occ_alpha_2
            occ_fw = length_sq(flow_diff_fw) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
            occ_bw = length_sq(flow_diff_bw) < occ_thresh
            # if IF_DEBUG:
            #     temp_ = sum_func(flow_diff_fw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_fw')
            #     temp_ = sum_func(flow_diff_bw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_bw')
            #     tools.check_tensor(data=mag_sq, name='check occlusion mask mag_sq')
            #     tools.check_tensor(data=occ_thresh, name='check occlusion mask occ_thresh')
            return occ_fw.float(), occ_bw.float()

        def _forward_warp_occ_check(self, flow_bw):
            raise ValueError('not implemented')

        @classmethod
        def torch_outgoing_occ_check(cls, flow):
            # out going pixels=0, others=1
            B, C, H, W = flow.size()
            # mesh grid
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
            flow_x, flow_y = torch.split(flow, 1, 1)
            if flow.is_cuda:
                xx = xx.cuda()
                yy = yy.cuda()
            # tools.check_tensor(flow_x, 'flow_x')
            # tools.check_tensor(flow_y, 'flow_y')
            # tools.check_tensor(xx, 'xx')
            # tools.check_tensor(yy, 'yy')
            pos_x = xx + flow_x
            pos_y = yy + flow_y
            # tools.check_tensor(pos_x, 'pos_x')
            # tools.check_tensor(pos_y, 'pos_y')
            # print(' ')
            # check mask
            outgoing_mask = torch.ones_like(pos_x)
            outgoing_mask[pos_x > W - 1] = 0
            outgoing_mask[pos_x < 0] = 0
            outgoing_mask[pos_y > H - 1] = 0
            outgoing_mask[pos_y < 0] = 0
            return outgoing_mask.float()

        @classmethod
        def torch_get_obj_occ_check(cls, occ_mask, out_occ):
            outgoing_mask = torch.zeros_like(occ_mask)
            if occ_mask.is_cuda:
                outgoing_mask = outgoing_mask.cuda()
            outgoing_mask[occ_mask == 1] = 1  # not occluded regions =1
            outgoing_mask[out_occ == 0] = 1  # 'out' regions=1, the rest regions=0(object moving inside the image plane)
            return outgoing_mask
    # Equivariant loss: here I call the self-supervision proposed in ARFlow as Equivariant loss
    class Eq_Loss_model(tools.abstract_config):
        def __init__(self, **kwargs):
            self.equivariant_loss_mask_norm = False

            self.equivariant_add_noise = False

            self.equivariant_hflip = True  # False
            self.equivariant_rotate = (-0.2, 0.2, -0.015, 0.015)  # [-0.01, 0.01, -0.01, 0.01]
            self.equivariant_squeeze = (0.86, 1.16, 1.0, 1.0)  # [1.0, 1.0, 1.0, 1.0]
            self.equivariant_trans = (0.2, 0.015)  # [0.04, 0.005]
            self.equivariant_vflip = False  # False
            self.equivariant_zoom = (1.0, 1.5, 0.985, 1.015)  # [1.0, 1.4, 0.99, 1.01]

            self.update(kwargs)

            class sp_conf():

                def __init__(self, conf):
                    self.add_noise = conf.equivariant_add_noise  # False
                    self.hflip = conf.equivariant_hflip  # False
                    self.rotate = conf.equivariant_rotate  # [-0.01, 0.01, -0.01, 0.01]
                    self.squeeze = conf.equivariant_squeeze  # [1.0, 1.0, 1.0, 1.0]
                    self.trans = conf.equivariant_trans  # [0.04, 0.005]
                    self.vflip = conf.equivariant_vflip  # False
                    self.zoom = conf.equivariant_zoom  # [1.0, 1.4, 0.99, 1.01]

            eq_conf = sp_conf(self)
            self.eq_transform = tensor_tools.SP_transform.RandomAffineFlow(eq_conf, addnoise=self.equivariant_add_noise).cuda()  #
            self.eq_transform.requires_grad = False

        def create_sample(self, im1, im2, flow_f, occ_f):
            flow_fw_pseudo_label, occ_fw_pseudo_label = flow_f.clone().detach(), occ_f.clone().detach()
            # spatial transform on aug images
            s = {'imgs': [im1, im2], 'flows_f': [flow_fw_pseudo_label], 'masks_f': [occ_fw_pseudo_label]}
            st_res = self.eq_transform(s)
            flow_t, noc_t = st_res['flows_f'][0], st_res['masks_f'][0]  # used as labels
            im1_st, im2_st = st_res['imgs']  # used as input
            return im1_st, im2_st, flow_t, noc_t

        def equivariant_loss_sequence(self, label, mask, flow_preds, gamma=0.8):
            n_predictions = len(flow_preds)
            flow_loss = 0.0
            for i in range(n_predictions):
                i_weight = gamma ** (n_predictions - i - 1)
                i_loss = (flow_preds[i] - label).abs()
                if self.equivariant_loss_mask_norm:
                    flow_loss += i_weight * ((mask * i_loss).mean() / (mask.mean() + 1e-6))
                else:
                    flow_loss += i_weight * (mask * i_loss).mean()
            return flow_loss

        def equivariant_loss(self, label, mask, flow_pred):
            i_loss = (flow_pred - label).abs()
            if self.equivariant_loss_mask_norm:
                flow_loss = (mask * i_loss).mean() / (mask.mean() + 1e-6)
            else:
                flow_loss = (mask * i_loss).mean()
            return flow_loss

    def __init__(self, **kwargs):
        self.photo_loss_weight = 1
        self.photo_loss_type = 'abs_robust'  # abs_robust, charbonnier,L1,SSIM(can not use)
        self.norm_photo_loss_scale = 0  # the original image scale is 0~255

        self.smooth_loss_type = 'edge'  # 'edge' or 'delta'
        self.smooth_loss_weight_1 = 0
        self.smooth_loss_weight_2 = 0
        self.smooth_loss_edge_constant = 150.0
        self.smooth_loss_edge_weight_type = 'exp'
        self.smooth_loss_edge_error_type = 'abs_robust'

        # occ check model
        self.occ_type = 'for_back_check'
        self.occ_alpha_1 = 1.0
        self.occ_alpha_2 = 0.05  # show area rate of the occlusion regions during training (print to log information) this is done
        self.obj_out_all = 'obj'  # obj, out, all, when boundary dilated warping is used, this should be 'obj'
        self.stop_occ_grad = False

        self.update(kwargs)
        self.occ_model = self.Occ_Check_Model(**self.to_dict)

    def _smooth_loss(self, im, flow):
        if self.smooth_loss_weight_1 <= 0 and self.smooth_loss_weight_2 <= 0:
            return None
        smooth_loss = 0
        if self.smooth_loss_weight_1 > 0:
            if self.smooth_loss_type == 'edge':
                smooth_loss += self.smooth_loss_weight_1 * self.edge_aware_smoothness_order1(img=im, pred=flow,
                                                                                             constant=self.smooth_loss_edge_constant,
                                                                                             weight_type=self.smooth_loss_edge_weight_type,
                                                                                             error_type=self.smooth_loss_edge_error_type)
            elif self.smooth_loss_type == 'delta':
                smooth_loss += self.smooth_loss_weight_1 * self.flow_smooth_delta(flow=flow, if_second_order=False)
            else:
                raise ValueError('wrong smooth_type: %s' % self.smooth_loss_type)

        # 2 order smooth loss
        if self.smooth_loss_weight_2 > 0:
            if self.smooth_loss_type == 'edge':
                smooth_loss += self.smooth_loss_weight_2 * self.edge_aware_smoothness_order2(img=im, pred=flow,
                                                                                             constant=self.smooth_loss_edge_constant,
                                                                                             weight_type=self.smooth_loss_edge_weight_type,
                                                                                             error_type=self.smooth_loss_edge_error_type)
            elif self.smooth_loss_type == 'delta':
                smooth_loss += self.smooth_loss_weight_2 * self.flow_smooth_delta(flow=flow, if_second_order=True)
            else:
                raise ValueError('wrong smooth_type: %s' % self.smooth_loss_type)
        return smooth_loss

    def __photometric_loss_no_occ(self, flow_f, im2_raw, im1_crop, crop_start):
        if self.norm_photo_loss_scale > 0:  # the original image scale is 0~255
            im1_crop, im2_raw = self._norm_photo(self.norm_photo_loss_scale, im1_crop, im2_raw)

        # ==== warp ==== boundary warp is used
        im1_warp = tensor_tools.torch_warp_boundary(im2_raw, flow_f, crop_start)  # warped im1 by forward flow and im2, you can also use nianjin warp here

        photo_loss = self.photo_loss_multi_type(im1_crop, im1_warp, occ_mask=None, photo_loss_type=self.photo_loss_type,
                                                photo_loss_use_occ=False)
        if self.photo_loss_weight > 0:
            photo_loss = photo_loss * self.photo_loss_weight
        return photo_loss

    def __photometric_loss_occ(self, flow_f, occ_f, im2_raw, im1_crop, crop_start):
        if self.norm_photo_loss_scale > 0:  # the original image scale is 0~255
            im1_crop, im2_raw = self._norm_photo(self.norm_photo_loss_scale, im1_crop, im2_raw)

        # ==== warp ==== boundary warp is used
        im1_warp = tensor_tools.torch_warp_boundary(im2_raw, flow_f, crop_start)  # warped im1 by forward flow and im2

        photo_loss = self.photo_loss_multi_type(im1_crop, im1_warp, occ_mask=occ_f, photo_loss_type=self.photo_loss_type,
                                                photo_loss_use_occ=True)
        if self.photo_loss_weight > 0:
            photo_loss = photo_loss * self.photo_loss_weight
        return photo_loss

    # single forward，only photo loss and smooth loss are computed
    def single_loss_photo_smooth(self, flow_pred, im1_crop, im2_raw, crop_start):
        """ Loss function defined over flow prediction """
        smooth_loss = self._smooth_loss(im1_crop, flow_pred)
        photo_loss = self.__photometric_loss_no_occ(flow_pred, im2_raw, im1_crop, crop_start)
        return photo_loss, smooth_loss

    # bi-directional flow，photo loss and smooth loss are computed，occlusion regions are checked from photo loss.
    def bidirection_photo_occ_smooth(self, flow_pred_f, flow_pred_b, im1_crop, im2_crop, im1_raw, im2_raw, crop_start):

        occ_f, occ_b = self.occ_model(flow_pred_f, flow_pred_b)
        if self.stop_occ_grad:
            occ_f = occ_f.clone().detach()
            occ_b = occ_b.clone().detach()
        occ_area = (torch.mean(occ_f) + torch.mean(occ_b)) / 2.0

        # compute loss function
        smooth_loss_f = self._smooth_loss(im1_crop, flow_pred_f)
        smooth_loss_b = self._smooth_loss(im2_crop, flow_pred_b)

        photo_loss_f = self.__photometric_loss_occ(flow_pred_f, occ_f, im2_raw, im1_crop, crop_start)
        photo_loss_b = self.__photometric_loss_occ(flow_pred_b, occ_b, im1_raw, im2_crop, crop_start)

        photo_loss = photo_loss_f + photo_loss_b
        if smooth_loss_f is None or smooth_loss_b is None:
            smooth_loss = None
        else:
            smooth_loss = smooth_loss_f + smooth_loss_b

        return photo_loss, smooth_loss, occ_area, occ_f, occ_b

    @classmethod
    def photo_loss_multi_type(cls, x, y, occ_mask=None, photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                              photo_loss_use_occ=False,
                              ):
        occ_weight = occ_mask
        if photo_loss_type == 'abs_robust':
            photo_diff = x - y
            loss_diff = (torch.abs(photo_diff) + 0.01).pow(0.4)
        elif photo_loss_type == 'charbonnier':
            photo_diff = x - y
            loss_diff = ((photo_diff) ** 2 + 1e-6).pow(0.4)
        elif photo_loss_type == 'L1':
            photo_diff = x - y
            loss_diff = torch.abs(photo_diff + 1e-6)
        elif photo_loss_type == 'SSIM':
            if occ_mask is None or not photo_loss_use_occ:
                raise ValueError('wrong photo_loss type: %s, SSIM need occ mask' % photo_loss_type)
            loss_diff, occ_weight = cls.weighted_ssim(x, y, occ_mask)
        else:
            raise ValueError('wrong photo_loss type: %s' % photo_loss_type)

        if photo_loss_use_occ:
            photo_loss = torch.sum(loss_diff * occ_weight) / (torch.sum(occ_weight) + 1e-6)
        else:
            photo_loss = torch.mean(loss_diff)
        return photo_loss

    @classmethod
    def weighted_ssim(cls, x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
        """Computes a weighted structured image similarity measure.
        Args:
          x: a batch of images, of shape [B, C, H, W].
          y:  a batch of images, of shape [B, C, H, W].
          weight: shape [B, 1, H, W], representing the weight of each
            pixel in both images when we come to calculate moments (means and
            correlations). values are in [0,1]
          c1: A floating point number, regularizes division by zero of the means.
          c2: A floating point number, regularizes division by zero of the second
            moments.
          weight_epsilon: A floating point number, used to regularize division by the
            weight.

        Returns:
          A tuple of two pytorch Tensors. First, of shape [B, C, H-2, W-2], is scalar
          similarity loss per pixel per channel, and the second, of shape
          [B, 1, H-2. W-2], is the average pooled `weight`. It is needed so that we
          know how much to weigh each pixel in the first tensor. For example, if
          `'weight` was very small in some area of the images, the first tensor will
          still assign a loss to these pixels, but we shouldn't take the result too
          seriously.
        """

        def _avg_pool3x3(x):
            # tf kernel [b,h,w,c]
            return F.avg_pool2d(x, (3, 3), (1, 1))
            # return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')

        if c1 == float('inf') and c2 == float('inf'):
            raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                             'likely unintended.')
        average_pooled_weight = _avg_pool3x3(weight)
        weight_plus_epsilon = weight + weight_epsilon
        inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

        def weighted_avg_pool3x3(z):
            wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
            return wighted_avg * inverse_average_pooled_weight

        mu_x = weighted_avg_pool3x3(x)
        mu_y = weighted_avg_pool3x3(y)
        sigma_x = weighted_avg_pool3x3(x ** 2) - mu_x ** 2
        sigma_y = weighted_avg_pool3x3(y ** 2) - mu_y ** 2
        sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
        if c1 == float('inf'):
            ssim_n = (2 * sigma_xy + c2)
            ssim_d = (sigma_x + sigma_y + c2)
        elif c2 == float('inf'):
            ssim_n = 2 * mu_x * mu_y + c1
            ssim_d = mu_x ** 2 + mu_y ** 2 + c1
        else:
            ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        result = ssim_n / ssim_d
        return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight

    @classmethod
    def edge_aware_smoothness_order1(cls, img, pred, constant=1.0, weight_type='gauss', error_type='L1'):
        def gradient_x(img):
            gx = img[:, :, :-1, :] - img[:, :, 1:, :]
            return gx

        def gradient_y(img):
            gy = img[:, :, :, :-1] - img[:, :, :, 1:]
            return gy

        def weight_fn(x):
            if weight_type == 'gauss':
                y = x ** 2
            elif weight_type == 'exp':
                y = torch.abs(x)
            else:
                raise ValueError('')
            return y

        def error_fn(x):
            if error_type == 'L1':
                y = torch.abs(x)
            elif error_type == 'abs_robust':
                y = (torch.abs(x) + 0.01).pow(0.4)
            else:
                raise ValueError('')
            return y

        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)

        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)

        weights_x = torch.exp(-torch.mean(weight_fn(constant * image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(weight_fn(constant * image_gradients_y), 1, keepdim=True))

        smoothness_x = error_fn(pred_gradients_x) * weights_x
        smoothness_y = error_fn(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    @classmethod
    def edge_aware_smoothness_order2(cls, img, pred, constant=1.0, weight_type='gauss', error_type='L1'):
        def gradient_x(img, stride=1):
            gx = img[:, :, :-stride, :] - img[:, :, stride:, :]
            return gx

        def gradient_y(img, stride=1):
            gy = img[:, :, :, :-stride] - img[:, :, :, stride:]
            return gy

        def weight_fn(x):
            if weight_type == 'gauss':
                y = x ** 2
            elif weight_type == 'exp':
                y = torch.abs(x)
            else:
                raise ValueError('')
            return y

        def error_fn(x):
            if error_type == 'L1':
                y = torch.abs(x)
            elif error_type == 'abs_robust':
                y = (torch.abs(x) + 0.01).pow(0.4)
            else:
                raise ValueError('')
            return y

        pred_gradients_x = gradient_x(pred)
        pred_gradients_xx = gradient_x(pred_gradients_x)
        pred_gradients_y = gradient_y(pred)
        pred_gradients_yy = gradient_y(pred_gradients_y)

        image_gradients_x = gradient_x(img, stride=2)
        image_gradients_y = gradient_y(img, stride=2)

        weights_x = torch.exp(-torch.mean(weight_fn(constant * image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(weight_fn(constant * image_gradients_y), 1, keepdim=True))

        smoothness_x = error_fn(pred_gradients_xx) * weights_x
        smoothness_y = error_fn(pred_gradients_yy) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    @classmethod
    def flow_smooth_delta(cls, flow, if_second_order=False):
        def gradient(x):
            D_dy = x[:, :, 1:] - x[:, :, :-1]
            D_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            return D_dx, D_dy

        dx, dy = gradient(flow)
        # dx2, dxdy = gradient(dx)
        # dydx, dy2 = gradient(dy)
        if if_second_order:
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            smooth_loss = dx.abs().mean() + dy.abs().mean() + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
        else:
            smooth_loss = dx.abs().mean() + dy.abs().mean()
        # smooth_loss = dx.abs().mean() + dy.abs().mean()  # + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
        # 暂时不上二阶的平滑损失，似乎加上以后就太猛了，无法降低photo loss TODO
        return smooth_loss

    @classmethod
    def _norm_photo(cls, scale, *args):
        def temp(a, s):
            b = 2.0 * a / 255.0 - 1.0  # -1~1
            c = b * s
            return c

        return [temp(i, scale) for i in args]


class train_irrpwc(tools.abstract_config):
    def __init__(self, **kwargs):
        self.name = 'irrpwc'  # name your experiment
        self.train_dir = '/data/Optical_Flow_all/train_unsup_irrpwc'
        self.if_show_training = False

        # ==== dataset
        self.train_set = 'flyingchairs'  # determines which dataset to use for training
        self.validation_sets = ['flyingchairs', ]  # validation sets
        self.eval_batch_size = 8

        # ==== model
        self.pretrain_path = None
        self.if_froze_pwc = False
        self.if_use_correlation_pytorch = True  # just for debug, the speed is too slow
        self.gpus = None

        # ==== training
        self.log_name = ''
        self.lr = 0.00002
        self.wdecay = 0.00005
        self.num_steps = None
        self.num_epochs = 2000
        self.batch_size = 6
        self.optim_gamma = 1
        self.epsilon = 1e-8

        self.add_noise = True
        self.print_every = 20
        self.eval_every = 500

        # ==== loss
        self.train_func = 'train_single_direction_photo_smooth'  # if use bidirectional flow estimation

        self.photo_loss_weight = 1
        self.norm_photo_loss_scale = 0  # the original image scale is 0~255, 0=don't change, 1=[-1,1],0.5=[-0.5,0.5],
        self.photo_loss_type = 'abs_robust'  # abs_robust, charbonnier,L1,SSIM(can not use)

        self.smooth_loss_type = 'edge'  # 'edge' or 'delta'
        self.smooth_loss_weight_1 = 0
        self.smooth_loss_weight_2 = 0
        self.smooth_loss_edge_constant = 150.0
        self.smooth_loss_edge_weight_type = 'exp'
        self.smooth_loss_edge_error_type = 'abs_robust'

        # occ check model
        self.occ_type = 'for_back_check'
        self.occ_alpha_1 = 1.0
        self.occ_alpha_2 = 0.05  #
        self.obj_out_all = 'obj'  # obj, out, all, when boundary dilated warping is used, this should be 'obj'
        self.stop_occ_grad = False

        # ==== test
        self.eval_save_results = False
        self.eval_some_save_results = False

        # === dataset aug augmentation params
        self.aug_sintel_final_prob = 0 # using final image as the photometric augmentation version of clean image
        self.aug_switch_prob = 0.5
        self.aug_crop_size = (320, 320)  # 1024x436 image
        self.aug_crop_rho = 8
        self.aug_horizontal_prob = 0.5
        # photo metric and occlusion aug
        self.aug_color_prob = 0  # do not use aug
        self.aug_color_asymmetric_prob = 0
        self.aug_eraser_prob = 0

        # ==== eq loss parameters
        self.equivariant_loss_mask_norm = False
        self.eq_loss_weight = 0.5
        self.equivariant_add_noise = False
        self.equivariant_hflip = True  # False
        self.equivariant_rotate = (-0.2, 0.2, -0.015, 0.015)  # [-0.01, 0.01, -0.01, 0.01]
        self.equivariant_squeeze = (0.86, 1.16, 1.0, 1.0)  # [1.0, 1.0, 1.0, 1.0]
        self.equivariant_trans = (0.2, 0.015)  # [0.04, 0.005]
        self.equivariant_vflip = False  # False
        self.equivariant_zoom = (1.0, 1.5, 0.985, 1.015)  # [1.0, 1.4, 0.99, 1.01]

        self.update(kwargs)
        self.datatype = 'base'
        self.show_training_dir = ''
        self._texer = tools.Text_img()

    def __call__(self):
        torch.manual_seed(1234)
        np.random.seed(1234)
        file_tools.check_dir(self.train_dir)
        if self.if_show_training:
            self.show_training_dir = os.path.join(self.train_dir, 'show_training')
            file_tools.check_dir(self.show_training_dir)
        self.num_steps = self.eval_every * self.num_epochs
        train_func_dict = {
            # 单向光流估计，只用了photo和smooth loss，没有使用occ，各个sequence算了无监督损失，乘以权重，再加起来
            'train_single_direction_photo_smooth': self.train_single_direction_photo_smooth,
            # 双向光流估计，只用了photo和smooth loss，没有使用occ
            'train_bi_direction_photo_smooth': self.train_bi_direction_photo_smooth,
            # 双向光流估计，只用了photo和smooth loss，使用occ
            'train_bi_direction_photo_occ_smooth': self.train_bi_direction_photo_occ_smooth,
            # 双向光流估计，photo smooth，加上occ check，接下来算eq loss
            'train_bi_direction_photo_occ_smooth_eqloss': self.train_bi_direction_photo_occ_smooth_eqloss,

        }
        if self.train_func in train_func_dict.keys():
            train_func = train_func_dict[self.train_func]
            train_func()
        else:
            raise ValueError('wrong train_func: %s' % self.train_func)

    def fetch_optimizer(self, model):
        """ Create the optimizer and learning rate scheduler """
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wdecay, eps=self.epsilon)

        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, self.lr, self.num_steps + 100,
        #                                           pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.optim_gamma)

        return optimizer, scheduler

    def fetch_dataloader(self):
        print('=' * 3 + ' build training dataset ' + '=' * 3)
        if self.train_set == 'flyingchairs':
            data_conf = UnFlowDataset.Config(data_name='flyingchairs', **self.to_dict)
            dataset = data_conf()
        elif self.train_set == 'sintel':
            data_conf = UnFlowDataset.Config(data_name='sintel', data_pass='clean', **self.to_dict)
            dataset = data_conf()
        elif self.train_set == 'KITTI':
            data_conf_12 = UnFlowDataset.Config(data_name='KITTI', data_pass='2012mv', **self.to_dict)
            k2012 = data_conf_12()
            data_conf_15 = UnFlowDataset.Config(data_name='KITTI', data_pass='2015mv', **self.to_dict)
            k2015 = data_conf_15()
            dataset = k2012 + k2015
        else:
            raise ValueError('wrong train set name')
        # res = {'im1': None, 'im2': None, 'im1_crop': None, 'im2_crop': None, 'name': None,
        #        'im1_crop_aug': None, 'im2_crop_aug': None, 'crop_start': None}
        train_loader = tools.data_prefetcher_dict(dataset, gpu_keys=('im1', 'im2', 'im1_crop', 'im2_crop', 'im1_crop_aug', 'im2_crop_aug', 'crop_start'),
                                                  batch_size=self.batch_size, pin_memory=False, shuffle=True,
                                                  num_workers=4, drop_last=True, gpu_opt=self.gpus)
        return train_loader

    def fetch_network(self):  # todo
        pass

    ''' the training code for different stage is divided. For example, you can first train using singloe_photo_smooth. 
    Then, load model and train using bi_direction_photo_smooth. then add occ checking and then add eqloss.'''

    def train_single_direction_photo_smooth(self):
        model = IRRPWC(self)
        loss_func = Unsup_Loss(**self.to_dict)
        if self.pretrain_path is not None:
            model.load_model(self.pretrain_path, if_relax=True)
        model = model.choose_gpu(self.gpus)
        print("Parameter Count: %.2f M " % (tensor_tools.count_parameters(model) / 1000 / 1000))
        train_loader = self.fetch_dataloader()
        # if self.train_set != 'flyingchairs':
        #     model.module.freeze_bn()
        tem = Training_eval_manager(**self.to_dict)
        optimizer, scheduler = self.fetch_optimizer(model)
        current_val, best_val, best_epoch = 0, 0, 0
        error_meter = tools.Avg_meter_ls()
        should_keep_training = True
        i_batch = 0
        epoch = 0
        timer = tools.TimeClock()
        timer.start()
        indict = {'if_single_forward': True, }
        while should_keep_training:
            data_blob = train_loader.next()
            i_batch += 1
            if data_blob is None:
                data_blob = train_loader.next()
                assert data_blob is not None
            model.train()
            optimizer.zero_grad()
            image1, image2 = data_blob['im1_crop_aug'], data_blob['im2_crop_aug']
            # if self.add_noise:
            #     stdv = np.random.uniform(0.0, 5.0)
            #     image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
            #     image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
            out_dict = model(image1, image2, indict)
            flow_f = out_dict['flow_f']

            photo_loss, smooth_loss = loss_func.single_loss_photo_smooth(flow_f, data_blob['im1_crop'], data_blob['im2'], data_blob['crop_start'])  # 俩损失已经乘过了权重
            loss = photo_loss
            if smooth_loss is not None:
                loss += smooth_loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            num = image1.shape[0]
            error_meter.update(name='photo_loss', val=photo_loss.item(), num=num, short_name='ph')
            if smooth_loss is not None:
                error_meter.update(name='smooth_loss', val=smooth_loss.item(), num=num, short_name='sm')

            if i_batch % self.print_every == 0:
                prt = error_meter.print_all_losses()
                print('%s%s|%s [%.2f|%s] %s' % (self.log_name, i_batch, epoch, best_val, best_epoch, prt))
            if i_batch % self.eval_every == 0:
                tmodel = Flow_test_model(flownet=model)
                save_info = {'iter': i_batch, 'epoch': epoch, 'train_loss': error_meter.print_all_losses_final()}
                current_val, best_val = tem(tmodel, save_name=self.name, save_info=save_info)  # get current val score and best val score
                if current_val == best_val:
                    best_epoch = epoch
                print(' ===epoch=%s  current score: %.2f, best score: %.2f(%s)' % (epoch, current_val, best_val, best_epoch))
                timer.end()
                print(' === epoch use time %.2f' % timer.get_during())
                # if self.train_set != 'flyingchairs':
                #     model.module.freeze_bn()
                if i_batch > self.num_steps:
                    should_keep_training = False
                    break
                epoch += 1
                timer.start()
                error_meter.reset()
                scheduler.step()

    def train_bi_direction_photo_smooth(self):
        model = IRRPWC(self)
        loss_func = Unsup_Loss(**self.to_dict)
        if self.pretrain_path is not None:
            model.load_model(self.pretrain_path, if_relax=True)
        model = model.choose_gpu(self.gpus)
        print("Parameter Count: %.2f M " % (tensor_tools.count_parameters(model) / 1000 / 1000))
        train_loader = self.fetch_dataloader()
        # if self.train_set != 'flyingchairs':
        #     model.module.freeze_bn()
        tem = Training_eval_manager(**self.to_dict)
        optimizer, scheduler = self.fetch_optimizer(model)
        current_val, best_val, best_epoch = 0, 0, 0
        error_meter = tools.Avg_meter_ls()
        should_keep_training = True
        i_batch = 0
        epoch = 0
        timer = tools.TimeClock()
        timer.start()
        indict = {'if_single_forward': False, }
        while should_keep_training:
            data_blob = train_loader.next()
            i_batch += 1
            if data_blob is None:
                data_blob = train_loader.next()
                assert data_blob is not None
            model.train()
            optimizer.zero_grad()
            image1, image2 = data_blob['im1_crop_aug'], data_blob['im2_crop_aug']
            # if self.add_noise:
            #     stdv = np.random.uniform(0.0, 5.0)
            #     image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
            #     image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
            out_dict = model(image1, image2, indict)
            flow_f = out_dict['flow_f']
            flow_b = out_dict['flow_b']

            photo_loss_f, smooth_loss_f = loss_func.single_loss_photo_smooth(flow_f, data_blob['im1_crop'], data_blob['im2'], data_blob['crop_start'])  # 俩损失已经乘过了权重
            photo_loss_b, smooth_loss_b = loss_func.single_loss_photo_smooth(flow_b, data_blob['im2_crop'], data_blob['im1'], data_blob['crop_start'])  # 俩损失已经乘过了权重
            photo_loss = photo_loss_f + photo_loss_b
            loss = photo_loss
            if smooth_loss_f is not None and smooth_loss_b is not None:
                smooth_loss = smooth_loss_f + smooth_loss_b
                loss += smooth_loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            num = image1.shape[0]
            error_meter.update(name='photo_loss', val=photo_loss.item(), num=num, short_name='ph')
            if smooth_loss_f is not None and smooth_loss_b is not None:
                error_meter.update(name='smooth_loss', val=smooth_loss.item(), num=num, short_name='sm')

            if i_batch % self.print_every == 0:
                prt = error_meter.print_all_losses()
                print('%s%s|%s [%.2f|%s] %s' % (self.log_name, i_batch, epoch, best_val, best_epoch, prt))
            if i_batch % self.eval_every == 0:
                tmodel = Flow_test_model(flownet=model)
                save_info = {'iter': i_batch, 'epoch': epoch, 'train_loss': error_meter.print_all_losses_final()}
                current_val, best_val = tem(tmodel, save_name=self.name, save_info=save_info)  # get current val score and best val score
                if current_val == best_val:
                    best_epoch = epoch
                print(' ===epoch=%s  current score: %.2f, best score: %.2f(%s)' % (epoch, current_val, best_val, best_epoch))
                timer.end()
                print(' === epoch use time %.2f' % timer.get_during())
                if self.train_set != 'flyingchairs':
                    model.module.freeze_bn()
                if i_batch > self.num_steps:
                    should_keep_training = False
                    break
                epoch += 1
                timer.start()
                error_meter.reset()
                scheduler.step()

    def train_bi_direction_photo_occ_smooth(self):
        model = IRRPWC(self)
        loss_func = Unsup_Loss(**self.to_dict)
        if self.pretrain_path is not None:
            model.load_model(self.pretrain_path, if_relax=True)
        model = model.choose_gpu(self.gpus)
        print("Parameter Count: %.2f M " % (tensor_tools.count_parameters(model) / 1000 / 1000))
        train_loader = self.fetch_dataloader()
        # if self.train_set != 'flyingchairs':
        #     model.module.freeze_bn()
        tem = Training_eval_manager(**self.to_dict)
        optimizer, scheduler = self.fetch_optimizer(model)
        current_val, best_val, best_epoch = 0, 0, 0
        error_meter = tools.Avg_meter_ls()
        should_keep_training = True
        i_batch = 0
        epoch = 0
        timer = tools.TimeClock()
        timer.start()
        indict = {'if_single_forward': False, }
        while should_keep_training:
            data_blob = train_loader.next()
            i_batch += 1
            if data_blob is None:
                data_blob = train_loader.next()
                assert data_blob is not None
            model.train()
            optimizer.zero_grad()
            image1, image2 = data_blob['im1_crop_aug'], data_blob['im2_crop_aug']
            # if self.add_noise:
            #     stdv = np.random.uniform(0.0, 5.0)
            #     image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
            #     image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            out_dict = model(image1, image2, indict)
            flow_f = out_dict['flow_f']
            flow_b = out_dict['flow_b']

            photo_loss, smooth_loss, occ_area, occ_f, occ_b = loss_func.bidirection_photo_occ_smooth(flow_pred_f=flow_f, flow_pred_b=flow_b,
                                                                                                     im1_crop=data_blob['im1_crop'], im2_crop=data_blob['im2_crop'],
                                                                                                     im1_raw=data_blob['im1'], im2_raw=data_blob['im2'],
                                                                                                     crop_start=data_blob['crop_start'])
            loss = photo_loss
            if smooth_loss is not None:
                loss += smooth_loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            num = image1.shape[0]
            error_meter.update(name='photo_loss', val=photo_loss.item(), num=num, short_name='ph')
            if smooth_loss is not None:
                error_meter.update(name='smooth_loss', val=smooth_loss.item(), num=num, short_name='sm')
            error_meter.update(name='occ_area', val=occ_area.item(), num=num)  # show occ area for choosing occ checking parameters

            if i_batch % self.print_every == 0:
                prt = error_meter.print_all_losses()
                print('%s%s|%s [%.2f|%s] %s' % (self.log_name, i_batch, epoch, best_val, best_epoch, prt))
                if self.if_show_training:
                    image1_show = self._check_show_im(image1, 'image1')
                    image2_show = self._check_show_im(image2, 'image2')

                    occ_f_show = self._check_show_im(occ_f, 'occ_f')
                    flow_f_show = self._check_show_im(flow_f, 'flow_pred_f', if_is_flow=True)

                    occ_b_show = self._check_show_im(occ_b, 'occ_b')
                    flow_b_show = self._check_show_im(flow_b, 'flow_pred_b', if_is_flow=True)

                    imshow0 = np.concatenate((image1_show, image2_show), axis=1)
                    imshow1 = np.concatenate((flow_f_show, occ_f_show), axis=1)
                    imshow2 = np.concatenate((flow_b_show, occ_b_show), axis=1)
                    imshow = np.concatenate((imshow0, imshow1, imshow2), axis=0)
                    cv2.imwrite(os.path.join(self.show_training_dir, 'iter_%s.png' % i_batch), imshow)

            if i_batch % self.eval_every == 0:
                tmodel = Flow_test_model(flownet=model)
                save_info = {'iter': i_batch, 'epoch': '%s[%s]' % (epoch, error_meter.print_avg_loss('occ_area')), 'train_loss': error_meter.print_all_losses_final()}
                current_val, best_val = tem(tmodel, save_name=self.name, save_info=save_info)  # get current val score and best val score
                if current_val == best_val:
                    best_epoch = epoch
                print(' ===epoch=%s  current score: %.2f, best score: %.2f(%s)' % (epoch, current_val, best_val, best_epoch))
                timer.end()
                print(' === epoch use time %.2f' % timer.get_during())
                if self.train_set != 'flyingchairs':
                    model.module.freeze_bn()
                if i_batch > self.num_steps:
                    should_keep_training = False
                    break
                epoch += 1
                timer.start()
                error_meter.reset()
                scheduler.step()

    def train_bi_direction_photo_occ_smooth_eqloss(self):
        model = IRRPWC(self)
        loss_func = Unsup_Loss(**self.to_dict)
        eq_loss_model = Unsup_Loss.Eq_Loss_model(**self.to_dict)
        if self.pretrain_path is not None:
            model.load_model(self.pretrain_path, if_relax=True)
        model = model.choose_gpu(self.gpus)
        print("Parameter Count: %.2f M " % (tensor_tools.count_parameters(model) / 1000 / 1000))
        train_loader = self.fetch_dataloader()
        # if self.train_set != 'flyingchairs':
        #     model.module.freeze_bn()
        tem = Training_eval_manager(**self.to_dict)
        optimizer, scheduler = self.fetch_optimizer(model)
        current_val, best_val, best_epoch = 0, 0, 0
        error_meter = tools.Avg_meter_ls()
        should_keep_training = True
        i_batch = 0
        epoch = 0
        timer = tools.TimeClock()
        timer.start()
        indict = {'if_single_forward': False, }
        while should_keep_training:
            data_blob = train_loader.next()
            i_batch += 1
            if data_blob is None:
                data_blob = train_loader.next()
                assert data_blob is not None
            model.train()
            optimizer.zero_grad()
            image1, image2 = data_blob['im1_crop'], data_blob['im2_crop']  # todo 重点: teacher 给原图
            # if self.add_noise:
            #     stdv = np.random.uniform(0.0, 5.0)
            #     image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
            #     image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
            out_dict = model(image1, image2, indict)
            flow_f = out_dict['flow_f']
            flow_b = out_dict['flow_b']

            photo_loss, smooth_loss, occ_area, occ_f, occ_b = loss_func.bidirection_photo_occ_smooth(flow_pred_f=flow_f, flow_pred_b=flow_b,
                                                                                                     im1_crop=data_blob['im1_crop'], im2_crop=data_blob['im2_crop'],
                                                                                                     im1_raw=data_blob['im1'], im2_raw=data_blob['im2'],
                                                                                                     crop_start=data_blob['crop_start'])
            loss = photo_loss
            if smooth_loss is not None:
                loss += smooth_loss

            eq_im1, eq_im2, eq_flow_gt, eq_mask = eq_loss_model.create_sample(im1=image1, im2=image2, flow_f=flow_f, occ_f=occ_f)
            flow_predictions_f_eq = model(eq_im1, eq_im2, indict)
            eq_loss = eq_loss_model.equivariant_loss(label=eq_flow_gt, mask=eq_mask, flow_pred=flow_predictions_f_eq)
            loss += self.eq_loss_weight * eq_loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            num = image1.shape[0]
            error_meter.update(name='photo_loss', val=photo_loss.item(), num=num, short_name='ph')
            if smooth_loss is not None:
                error_meter.update(name='smooth_loss', val=smooth_loss.item(), num=num, short_name='sm')
            error_meter.update(name='eq_loss', val=eq_loss.item(), num=num)
            error_meter.update(name='occ_area', val=occ_area.item(), num=num)  # show occ area for choosing occ checking parameters

            if i_batch % self.print_every == 0:
                prt = error_meter.print_all_losses()
                print('%s%s|%s [%.2f|%s] %s' % (self.log_name, i_batch, epoch, best_val, best_epoch, prt))
                if self.if_show_training:
                    image1_show = self._check_show_im(image1, 'image1')
                    image2_show = self._check_show_im(image2, 'image2')
                    occ_f_show = self._check_show_im(occ_f, 'occ_f')
                    flow_show = self._check_show_im(flow_f, 'flow_pred', if_is_flow=True)

                    eq_im1_show = self._check_show_im(eq_im1, 'eq_im1')
                    eq_im2_show = self._check_show_im(eq_im2, 'eq_im2')
                    eq_flow_gt_show = self._check_show_im(eq_flow_gt, 'eq_flow_gt', if_is_flow=True)
                    eq_mask_show = self._check_show_im(eq_mask, 'eq_mask')
                    eq_flow_show = self._check_show_im(flow_predictions_f_eq[-1], 'eq_flow_pred', if_is_flow=True)
                    imshow0 = np.concatenate((image1_show, image2_show, occ_f_show), axis=1)
                    imshow1 = np.concatenate((flow_show, eq_im1_show, eq_im2_show), axis=1)
                    imshow2 = np.concatenate((eq_flow_gt_show, eq_mask_show, eq_flow_show), axis=1)
                    imshow = np.concatenate((imshow0, imshow1, imshow2), axis=0)
                    cv2.imwrite(os.path.join(self.show_training_dir, 'iter_%s.png' % i_batch), imshow)

            if i_batch % self.eval_every == 0:
                tmodel = Flow_test_model(flownet=model)
                save_info = {'iter': i_batch, 'epoch': '%s[%s]' % (epoch, error_meter.print_avg_loss('occ_area')), 'train_loss': error_meter.print_all_losses_final()}
                current_val, best_val = tem(tmodel, save_name=self.name, save_info=save_info)  # get current val score and best val score
                if current_val == best_val:
                    best_epoch = epoch
                print(' ===epoch=%s  current score: %.2f, best score: %.2f(%s)' % (epoch, current_val, best_val, best_epoch))
                timer.end()
                print(' === epoch use time %.2f' % timer.get_during())
                if self.train_set != 'flyingchairs':
                    model.module.freeze_bn()
                if i_batch > self.num_steps:
                    should_keep_training = False
                    break
                epoch += 1
                timer.start()
                error_meter.reset()
                scheduler.step()

    def _check_show_im(self, img, name='', if_is_flow=False):
        # tensor_tools.check_tensor(img, name)
        img_show = img[0].permute(1, 2, 0).detach().cpu().numpy()
        if if_is_flow:
            img_show = tensor_tools.flow_to_image_dmax(img_show)  # uint8 RGB image
        else:
            img_show = np.squeeze(img_show)
            if len(img_show.shape) == 2:  # this is a mask
                img_show = np.stack((img_show, img_show, img_show), 2)
            img_show = tensor_tools.im_norm(img_show)  # uint8 RGB image
        img_show = self._texer.put_text(img_show, name)
        # tensor_tools.check_tensor_np(img_show, name + '_np')
        return img_show

    def eval(self):
        model = IRRPWC(self)
        if self.pretrain_path is not None:
            model.load_model(self.pretrain_path, if_relax=False)
        model = model.choose_gpu(self.gpus)
        print("Parameter Count: %.2f M " % (tensor_tools.count_parameters(model) / 1000 / 1000))
        tem = Training_eval_manager(if_print_process=self.eval_save_results, **self.to_dict)
        tmodel = Flow_test_model(flownet=model)
        if self.eval_save_results:
            tmodel.do_save_results(result_save_dir=self.train_dir, some_save_results=self.eval_some_save_results)
        else:
            tmodel.do_save_results(result_save_dir=None, some_save_results=False)  # none = do not save results
        current_val, results = tem.eval(tmodel)
        print('score=%.2f' % current_val, results)

    @classmethod
    def demo(cls,):
        p = {
            'train_dir': '/data/Optical_Flow_all/train_unsuper_optical_flow/chairs_OIFlow_irrpwc',
            'if_show_training': False,
            # ==== dataset
            'train_set': 'flyingchairs',
            'validation_sets': ('flyingchairs',),
            'eval_batch_size': 8,
            'image_size': (384, 512),
            'datatype': 'base',
            # ==== model
            'pretrain_path': None,
            'if_froze_pwc': False,
            'if_use_correlation_pytorch': False,  # just for debug
            # ==== training
            'lr': 1e-4,
            'wdecay': 1e-5,
            'num_epochs': 2000,
            'batch_size': 6,
            'optim_gamma': 1,
            'print_every': 50,
            'eval_every': 500,
            # ==== loss
            'train_func': 'train_single_direction_photo_smooth',
            'photo_loss_weight': 1,
            'norm_photo_loss_scale': 0.5,  # the original image scale is 0~255
            'photo_loss_type': 'abs_robust',  # abs_robust, charbonnier,L1,SSIM(can not use)
            'smooth_loss_type': 'delta',  # 'edge' or 'delta'
            'smooth_loss_weight_1': 0.01,
            'smooth_loss_weight_2': 0,
            'occ_alpha_1': 0.05,
            'occ_alpha_2': 0.5,
            'obj_out_all': 'obj',  # obj, out, all, when boundary dilated warping is used, this should be 'obj'
            'stop_occ_grad': True,

            # ==== dataset aug
            'aug_sintel_final_prob': 0,
            'aug_crop_size': (320, 320),
            'aug_switch_prob': 0.5,
            'aug_horizontal_prob': 0.5,
            # photo metric and occlusion aug
            'aug_color_prob': 0,
            'aug_color_asymmetric_prob': 0,
            'aug_eraser_prob': 0,
            # ==== test
            'eval_save_results': False,
            'eval_some_save_results': False,
            # ==== eq loss parameters
            'equivariant_loss_mask_norm': False,
            'eq_loss_weight': 0.1,
            'equivariant_add_noise': False,
            'equivariant_hflip': True,
            'equivariant_rotate': (-0.2, 0.2, -0.015, 0.015),  # [-0.01, 0.01, -0.01, 0.01]
            'equivariant_squeeze': (0.86, 1.16, 1.0, 1.0),  # [1.0, 1.0, 1.0, 1.0]
            'equivariant_trans': (0.2, 0.015),  # [0.04, 0.005]
            'equivariant_vflip': False,
            'equivariant_zoom': (1.0, 1.5, 0.985, 1.015),  # [1.0, 1.4, 0.99, 1.01]
        }
        trainer = train_irrpwc(**p)
        trainer()
