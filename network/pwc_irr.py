import torch
import torch.nn as nn
import torch.nn.functional as F
from network.correlation_package.correlation import Correlation
from utils.tools import tools, file_tools, tensor_tools
from utils.correlation_pytorch import CorrTorch_lkm
import cv2
import os
import numpy as np


class Network_tool():
    class WarpingLayer_no_div(tools.abstract_model):

        def __init__(self):
            super(Network_tool.WarpingLayer_no_div, self).__init__()

        def forward(self, x, flow):
            B, C, H, W = x.size()
            # mesh grid
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()
            if x.is_cuda:
                grid = grid.cuda()
            # print(grid.shape,flo.shape,'...')
            vgrid = grid + flow
            # scale grid to [-1,1]
            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
            vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
            x_warp = F.grid_sample(x, vgrid, padding_mode='zeros')
            if x.is_cuda:
                mask = torch.ones(x.size(), requires_grad=False).cuda()
            else:
                mask = torch.ones(x.size(), requires_grad=False)  # .cuda()
            mask = F.grid_sample(mask, vgrid)
            mask = (mask >= 1.0).float()
            return x_warp * mask

    class WarpingLayer(nn.Module):
        def __init__(self):
            super(Network_tool.WarpingLayer, self).__init__()

        def forward(self, x, flow, height_im, width_im, div_flow):
            flo_list = []
            flo_w = flow[:, 0] * 2 / max(width_im - 1, 1) / div_flow
            flo_h = flow[:, 1] * 2 / max(height_im - 1, 1) / div_flow
            flo_list.append(flo_w)
            flo_list.append(flo_h)
            flow_for_grid = torch.stack(flo_list).transpose(0, 1)
            grid = torch.add(self.get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)
            x_warp = F.grid_sample(x, grid)

            mask = torch.ones(x.size(), requires_grad=False).cuda()
            mask = F.grid_sample(mask, grid)
            mask = (mask >= 1.0).float()

            return x_warp * mask

        def get_grid(self, x):
            grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
            grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
            grid = torch.cat([grid_H, grid_V], 1)
            grids_cuda = grid.float().requires_grad_(False).cuda()
            return grids_cuda

    class FlowEstimatorDense(tools.abstract_model):

        def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32), out_channel=2):
            super(Network_tool.FlowEstimatorDense, self).__init__()
            N = 0
            ind = 0
            N += ch_in
            self.conv1 = Network_tool.conv(N, f_channels[ind])
            N += f_channels[ind]

            ind += 1
            self.conv2 = Network_tool.conv(N, f_channels[ind])
            N += f_channels[ind]

            ind += 1
            self.conv3 = Network_tool.conv(N, f_channels[ind])
            N += f_channels[ind]

            ind += 1
            self.conv4 = Network_tool.conv(N, f_channels[ind])
            N += f_channels[ind]

            ind += 1
            self.conv5 = Network_tool.conv(N, f_channels[ind])
            N += f_channels[ind]
            self.n_channels = N
            ind += 1
            self.conv_last = Network_tool.conv(N, out_channel, isReLU=False)

        def forward(self, x):
            x1 = torch.cat([self.conv1(x), x], dim=1)
            x2 = torch.cat([self.conv2(x1), x1], dim=1)
            x3 = torch.cat([self.conv3(x2), x2], dim=1)
            x4 = torch.cat([self.conv4(x3), x3], dim=1)
            x5 = torch.cat([self.conv5(x4), x4], dim=1)
            x_out = self.conv_last(x5)
            return x5, x_out

    class ContextNetwork(tools.abstract_model):

        def __init__(self, ch_in, f_channels=(128, 128, 128, 96, 64, 32, 2)):
            super(Network_tool.ContextNetwork, self).__init__()

            self.convs = nn.Sequential(
                Network_tool.conv(ch_in, f_channels[0], 3, 1, 1),
                Network_tool.conv(f_channels[0], f_channels[1], 3, 1, 2),
                Network_tool.conv(f_channels[1], f_channels[2], 3, 1, 4),
                Network_tool.conv(f_channels[2], f_channels[3], 3, 1, 8),
                Network_tool.conv(f_channels[3], f_channels[4], 3, 1, 16),
                Network_tool.conv(f_channels[4], f_channels[5], 3, 1, 1),
                Network_tool.conv(f_channels[5], f_channels[6], isReLU=False)
            )

        def forward(self, x):
            return self.convs(x)

    class FeatureExtractor(tools.abstract_model):

        def __init__(self, num_chs, if_end_relu=True, if_end_norm=False):
            super(Network_tool.FeatureExtractor, self).__init__()
            self.num_chs = num_chs
            self.convs = nn.ModuleList()

            for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
                layer = nn.Sequential(
                    Network_tool.conv(ch_in, ch_out, stride=2),
                    Network_tool.conv(ch_out, ch_out, isReLU=if_end_relu, if_IN=if_end_norm)
                )
                self.convs.append(layer)

        def forward(self, x):
            feature_pyramid = []
            for conv in self.convs:
                x = conv(x)
                feature_pyramid.append(x)

            return feature_pyramid[::-1]

    @classmethod
    def initialize_msra(cls, modules):
        print("Initializing MSRA")
        for layer in modules:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.LeakyReLU):
                pass

            elif isinstance(layer, nn.Sequential):
                pass

    @classmethod
    def conv(cls, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, if_IN=False, IN_affine=False, if_BN=False):
        if isReLU:
            if if_IN:
                return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                              padding=((kernel_size - 1) * dilation) // 2, bias=True),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.InstanceNorm2d(out_planes, affine=IN_affine)
                )
            elif if_BN:
                return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                              padding=((kernel_size - 1) * dilation) // 2, bias=True),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.BatchNorm2d(out_planes, affine=IN_affine)
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                              padding=((kernel_size - 1) * dilation) // 2, bias=True),
                    nn.LeakyReLU(0.1, inplace=True)
                )
        else:
            if if_IN:
                return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                              padding=((kernel_size - 1) * dilation) // 2, bias=True),
                    nn.InstanceNorm2d(out_planes, affine=IN_affine)
                )
            elif if_BN:
                return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                              padding=((kernel_size - 1) * dilation) // 2, bias=True),
                    nn.BatchNorm2d(out_planes, affine=IN_affine)
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                              padding=((kernel_size - 1) * dilation) // 2, bias=True)
                )

    @classmethod
    def upsample2d_flow_as(cls, inputs, target_as, mode="bilinear", if_rate=False):
        _, _, h, w = target_as.size()
        res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
        if if_rate:
            _, _, h_, w_ = inputs.size()
            # inputs[:, 0, :, :] *= (w / w_)
            # inputs[:, 1, :, :] *= (h / h_)
            u_scale = (w / w_)
            v_scale = (h / h_)
            u, v = res.chunk(2, dim=1)
            u *= u_scale
            v *= v_scale
            res = torch.cat([u, v], dim=1)
        return res

    @classmethod
    def upsample2d_as(cls, inputs, target_as, mode="bilinear"):
        _, _, h, w = target_as.size()
        return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)


class abs_flow_net(tools.abstract_model):
    def __init__(self):
        super(abs_flow_net, self).__init__()
        self.result_save_dir = None
        self.img_text_drawer = tools.Text_img(font='simplex', my_font_type='black_white')

    def post_img(self, *args, if_norm=False):
        ''' input numpy image: h,w,3'''

        def temp_func(a):
            if a.if_cuda:
                img_np = a[0].permute(1, 2, 0).cpu().numpy()
            else:
                img_np = a[0].permute(1, 2, 0).numpy()
            if if_norm:
                img_np = tensor_tools.im_norm(img_np)
            return img_np

        res = [temp_func(i) for i in args]
        return res

    def save_image(self, image_tensor, save_name='image', puttext=None):
        def tensor_to_np_for_save(a):
            b = tensor_tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        if self.result_save_dir is None:
            return
        image_tensor_np = tensor_to_np_for_save(image_tensor)
        show_im = tensor_tools.im_norm(image_tensor_np)[:, :, ::-1]
        show_im = self.img_text_drawer.put_text(show_im, puttext, scale=1)
        save_path = os.path.join(self.result_save_dir, '%s.png' % save_name)
        cv2.imwrite(save_path, show_im)

    def save_flow(self, flow_tensor, save_name='flow', puttext=None):
        def tensor_to_np_for_save(a):
            b = tensor_tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        if self.result_save_dir is None:
            return
            # print(self.save_running_process_dir, 'save flow %s' % name)
        flow_tensor_np = tensor_to_np_for_save(flow_tensor)
        save_path = os.path.join(self.result_save_dir, '%s.png' % save_name)
        flow_im = tensor_tools.flow_to_image_ndmax(flow_tensor_np)[:, :, ::-1]
        flow_im = self.img_text_drawer.put_text(flow_im, puttext, scale=1)
        cv2.imwrite(save_path, flow_im)

    def save_mask(self, image_tensor, save_name='mask', puttext=None):
        def tensor_to_np_for_save(a):
            b = tensor_tools.tensor_gpu(a, check_on=False)[0]
            c = b[0, :, :, :]
            c = np.transpose(c, (1, 2, 0))
            return c

        if self.result_save_dir is None:
            return
        image_tensor_np = tensor_to_np_for_save(image_tensor)
        show_im = tensor_tools.im_norm(image_tensor_np)
        if show_im.shape[2] == 1:
            show_im = np.concatenate((show_im, show_im, show_im), 2)
        show_im = self.img_text_drawer.put_text(show_im, puttext, scale=1)
        save_path = os.path.join(self.result_save_dir, '%s.png' % save_name)
        cv2.imwrite(save_path, show_im)


class IRRPWC(abs_flow_net):
    class Config(tools.abstract_config):
        def __init__(self, **kwargs):
            self.if_froze_pwc = False
            self.if_use_correlation_pytorch = False  # cpu debug only
            self.update(kwargs)

        def __call__(self, ):
            model = IRRPWC(self)
            return model

    def __init__(self, conf: Config):
        super(IRRPWC, self).__init__()
        self.conf = conf
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.num_chs = [3, 16, 32, 64, 96, 128, 192]
        #                  1/2 1/4 1/8 1/16 1/32 1/64
        self.lkm_correlation = CorrTorch_lkm(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1)  # correlation in pytorch

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2
        self.warping_layer = Network_tool.WarpingLayer_no_div()

        # ==================== build main block
        out_channel = 2

        self.estimator_f_channels = (128, 128, 96, 64, 32)
        self.context_f_channels = (128, 128, 128, 96, 64, 64, out_channel)
        self.conv_1x1 = nn.ModuleList([Network_tool.conv(192, 32, kernel_size=1, stride=1, dilation=1),
                                       Network_tool.conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       Network_tool.conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       Network_tool.conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       Network_tool.conv(32, 32, kernel_size=1, stride=1, dilation=1)])
        self.feature_pyramid_extractor = Network_tool.FeatureExtractor(self.num_chs)

        self.flow_estimators = Network_tool.FlowEstimatorDense(self.num_ch_in, f_channels=self.estimator_f_channels, out_channel=out_channel)  # produce feature
        self.context_networks = Network_tool.ContextNetwork(self.flow_estimators.n_channels + 2, f_channels=self.context_f_channels)
        # =======================  init params
        # Network_tool.initialize_msra(self.modules())
        if self.conf.if_froze_pwc:
            self._froze_PWC()

    def forward(self, im1, im2, input_dict: dict):
        if_single_forward = input_dict['if_single_forward'] if 'if_single_forward' in input_dict.keys() else False
        '''  ======================================  forward  =====================================  '''
        if if_single_forward:
            flow_f_pwc_out, flows = self._forward_2_frame_single(im1, im2)
            out_dict = {'flow_f': flow_f_pwc_out, 'flows': flows}
        else:
            flow_f_pwc_out, flow_b_pwc_out, flows = self._forward_2_frame(im1, im2)

            out_dict = {'flow_f': flow_f_pwc_out, 'flow_b': flow_b_pwc_out, 'flows': flows}
        return out_dict

    # ======== network forward and decoding
    def _forward_2_frame(self, x1_raw, x2_raw):
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid, x2_pyramid = self._pyramid_encode_layer(x1_raw=x1_raw, x2_raw=x2_raw)
        flows = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        # build pyramid
        feature_level_ls = []
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            x1_1by1 = self.conv_1x1[l](x1)
            x2_1by1 = self.conv_1x1[l](x2)
            feature_level_ls.append((x1, x1_1by1, x2, x2_1by1))
            if l == self.output_level:
                break
        for level, (x1, x1_1by1, x2, x2_1by1) in enumerate(feature_level_ls):
            flow_f, flow_b, flow_f_res, flow_b_res = self._decode_level_res(level=level, flow_1=flow_f, flow_2=flow_b, feature_1=x1, feature_1_1x1=x1_1by1, feature_2=x2,
                                                                            feature_2_1x1=x2_1by1)
            flow_f = flow_f + flow_f_res
            flow_b = flow_b + flow_b_res
            flows.append([flow_f, flow_b])
        flow_f_out = self._upsample_layer(flow_1_lr=flow_f, target_hr=x1_raw)
        flow_b_out = self._upsample_layer(flow_1_lr=flow_b, target_hr=x1_raw)
        return flow_f_out, flow_b_out, flows[::-1]

    def _decode_level_res(self, level, flow_1, flow_2, feature_1, feature_1_1x1, feature_2, feature_2_1x1):
        flow_1_up_bilinear = self._upsample_layer(flow_1_lr=flow_1, target_hr=feature_1)
        flow_2_up_bilinear = self._upsample_layer(flow_1_lr=flow_2, target_hr=feature_2)
        # warping
        if level == 0:
            feature_2_warp = feature_2
            feature_1_warp = feature_1
        else:
            feature_2_warp = self.warping_layer(feature_2, flow_1_up_bilinear)
            feature_1_warp = self.warping_layer(feature_1, flow_2_up_bilinear)
        # correlation
        if self.conf.if_use_correlation_pytorch:
            out_corr_1 = self.lkm_correlation(feature_1, feature_2_warp)
            out_corr_2 = self.lkm_correlation(feature_2, feature_1_warp)
        else:
            out_corr_1 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_1, feature_2_warp)
            out_corr_2 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_2, feature_1_warp)
        out_corr_relu_1 = self.leakyRELU(out_corr_1)
        out_corr_relu_2 = self.leakyRELU(out_corr_2)
        feature_int_1, flow_res_1 = self.flow_estimators(torch.cat([out_corr_relu_1, feature_1_1x1, flow_1_up_bilinear], dim=1))
        feature_int_2, flow_res_2 = self.flow_estimators(torch.cat([out_corr_relu_2, feature_2_1x1, flow_2_up_bilinear], dim=1))
        flow_1_up_bilinear_ = flow_1_up_bilinear + flow_res_1
        flow_2_up_bilinear_ = flow_2_up_bilinear + flow_res_2
        flow_fine_1 = self.context_networks(torch.cat([feature_int_1, flow_1_up_bilinear_], dim=1))
        flow_fine_2 = self.context_networks(torch.cat([feature_int_2, flow_2_up_bilinear_], dim=1))
        flow_1_res = flow_res_1 + flow_fine_1
        flow_2_res = flow_res_2 + flow_fine_2
        return flow_1_up_bilinear, flow_2_up_bilinear, flow_1_res, flow_2_res

    # ======== single forward
    def _forward_2_frame_single(self, x1_raw, x2_raw):
        _, _, height_im, width_im = x1_raw.size()
        # on the bottom level are original images
        x1_pyramid, x2_pyramid = self._pyramid_encode_layer(x1_raw=x1_raw, x2_raw=x2_raw)
        flows = []
        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        # build pyramid
        feature_level_ls = []
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            x1_1by1 = self.conv_1x1[l](x1)
            feature_level_ls.append((x1, x1_1by1, x2))
            if l == self.output_level:
                break
        for level, (x1, x1_1by1, x2) in enumerate(feature_level_ls):
            flow_f, flow_f_res = self._decode_level_res_single(level=level, flow_1=flow_f, feature_1=x1, feature_1_1x1=x1_1by1, feature_2=x2)
            flow_f = flow_f + flow_f_res
            flow_f_ = self._upsample_layer(flow_1_lr=flow_f, target_hr=x1_raw)
            flows.append(flow_f_)
        flow_f_out = self._upsample_layer(flow_1_lr=flow_f, target_hr=x1_raw)
        flows.append(flow_f_out)
        return flow_f_out, flows

    def _decode_level_res_single(self, level, flow_1, feature_1, feature_1_1x1, feature_2):
        flow_1_up_bilinear = self._upsample_layer(flow_1_lr=flow_1, target_hr=feature_1)
        # warping
        if level == 0:
            feature_2_warp = feature_2
            feature_1_warp = feature_1
        else:
            feature_2_warp = self.warping_layer(feature_2, flow_1_up_bilinear)
        # correlation
        if self.conf.if_use_correlation_pytorch:
            out_corr_1 = self.lkm_correlation(feature_1, feature_2_warp)
        else:
            out_corr_1 = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(feature_1, feature_2_warp)
        out_corr_relu_1 = self.leakyRELU(out_corr_1)
        feature_int_1, flow_res_1 = self.flow_estimators(torch.cat([out_corr_relu_1, feature_1_1x1, flow_1_up_bilinear], dim=1))
        flow_1_up_bilinear_ = flow_1_up_bilinear + flow_res_1
        flow_fine_1 = self.context_networks(torch.cat([feature_int_1, flow_1_up_bilinear_], dim=1))
        flow_1_res = flow_res_1 + flow_fine_1
        return flow_1_up_bilinear, flow_1_res

    # ======== functions
    def _pyramid_encode_layer(self, x1_raw, x2_raw):
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        return x1_pyramid, x2_pyramid

    def _upsample_layer(self, flow_1_lr, target_hr):
        flow_1_sr_bilinear = Network_tool.upsample2d_flow_as(flow_1_lr, target_hr, mode="bilinear", if_rate=True)
        return flow_1_sr_bilinear

    def _froze_PWC(self):
        for param in self.feature_pyramid_extractor.parameters():
            param.requires_grad = False
        for param in self.flow_estimators.parameters():
            param.requires_grad = False
        for param in self.context_networks.parameters():
            param.requires_grad = False
        for param in self.conv_1x1.parameters():
            param.requires_grad = False

    @classmethod
    def demo(cls):
        conf = IRRPWC.Config(if_froze_pwc=False, if_use_correlation_pytorch=True)
        net = conf()
        net.eval()

        im = np.zeros((1, 3, 240, 240))
        im = torch.from_numpy(im).float()

        out = net(im, im, {'if_single_forward': False})
        flow_f = out['flow_f']
        flow_b = out['flow_b']
        flows = out['flows']

        tensor_tools.check_tensor(flow_f, 'flowf')
        tensor_tools.check_tensor(flow_b, 'flow_b')
        for ind, (data1, data2) in enumerate(flows):
            tensor_tools.check_tensor(data1, '%s_1' % ind)
            tensor_tools.check_tensor(data2, '%s_1' % ind)


class Flow_test_model(tools.abs_test_model):
    def __init__(self, flownet: abs_flow_net):
        super(Flow_test_model, self).__init__()
        self.model = flownet
        self.model.eval()
        self.if_data_parallel = self.check_data_parallel()

        # save results
        self.result_save_dir = None
        self.some_save_results = False
        self.some_ids = []  # only save some results
        self.id_cnt = -1
        self.id_cnt_save_dir = None
        self.eval_id_scores = {}

    def check_data_parallel(self):
        name_dataparallel = torch.nn.DataParallel.__name__
        if type(self.model).__name__ == name_dataparallel:
            return True
        else:
            return False

    def eval_forward(self, im1, im2, *args, **kwargs):
        self.id_cnt += 1
        if self.result_save_dir is not None:
            self.id_cnt_save_dir = os.path.join(self.result_save_dir, '%s' % self.id_cnt)
            if self.some_save_results and self.id_cnt not in self.some_ids:
                flow_pr = im1.narrow(1, 0, 2)  # narrow: dim, begin channel, num channel
                flow_pr = torch.zeros_like(flow_pr)
                return flow_pr
            else:
                file_tools.check_dir(self.id_cnt_save_dir)
                abs_flow_net.set_result_save_dir(model=self.model, result_save_dir=self.id_cnt_save_dir)

        input_dict = kwargs
        input_dict.update({'if_loss': False})
        if im1.shape[0] == 1 and self.if_data_parallel:  # data parallel, batchsize=1, error on gpu1
            im1_, im2_ = torch.cat((im1, im1), dim=0), torch.cat((im2, im2), dim=0)
            flag = True
        else:
            im1_, im2_ = im1, im2
            flag = False

        out = self.model(im1_, im2_, input_dict)
        flow_pr = out['flow_f']
        if flag:
            flow_pr = flow_pr.narrow(0, 0, 1)  # narrow: dim, begin channel, num channel
        return flow_pr

    def save_model(self, save_path):
        tools.abstract_model.save_model_gpu(self.model, path=save_path)

    def do_save_results(self, result_save_dir=None, some_save_results=False):
        self.result_save_dir = result_save_dir
        self.some_save_results = some_save_results
        # define some id
        if result_save_dir is not None:
            self.some_ids = [7 * 7 * i + 1 for i in range(24)]  # [1,50,99,148, ..., 981, 1030, 1079, 1128]


if __name__ == '__main__':
    IRRPWC.demo()
