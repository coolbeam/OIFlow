import os
from utils.tools import tools, tensor_tools, file_tools, frame_utils
import random
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import zipfile
from glob import glob
from torchvision import transforms as vision_transforms
import imageio


class flyingthings(tools.abs_database):
    def __init__(self, only_image=False, data_pass='clean', **kwargs):
        super(flyingthings, self).__init__()
        self.only_image = only_image
        self.data_pass = data_pass
        self.root = '/data/Optical_Flow_all/datasets/flyingthings'
        self.dstype = {'clean': 'frames_cleanpass', 'final': 'frames_finalpass'}
        current_dir = os.path.split(os.path.realpath(__file__))[0]
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(os.path.join(self.root, self.dstype[self.data_pass], 'TRAIN/*/*')))
                image_dirs = sorted([os.path.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(os.path.join(self.root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([os.path.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(os.path.join(idir, '*.png')))
                    flows = sorted(glob(os.path.join(fdir, '*.pfm')))
                    for i in range(len(flows) - 1):
                        if direction == 'into_future':
                            sample = {'flow_path': flows[i], 'img1_path': images[i], 'img2_path': images[i + 1]}
                            # self.image_list += [[images[i], images[i + 1]]]
                            # self.flow_list += [flows[i]]
                            self.data_ls['train'].append(sample)
                        elif direction == 'into_past':
                            sample = {'flow_path': flows[i + 1], 'img1_path': images[i + 1], 'img2_path': images[i]}
                            # self.image_list += [[images[i + 1], images[i]]]
                            # self.flow_list += [flows[i + 1]]
                            self.data_ls['train'].append(sample)
        self._init_len()

    # return {'im1': img1, 'im2': img2, 'flow': flow, 'valid': valid, 'name': name}
    def sample(self, index, split):
        sample = self.data_ls[split][index % self.len[split]]

        img1 = frame_utils.read_gen(sample['img1_path'])
        img2 = frame_utils.read_gen(sample['img2_path'])
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
        res = {'im1': img1, 'im2': img2, 'name': os.path.split(sample['img1_path'])[1]}
        if self.only_image:
            return res
        flow = frame_utils.read_gen(sample['flow_path'])
        flow = np.array(flow).astype(np.float32)
        valid = None
        res.update({'flow': flow, 'valid': valid})
        return res

    @classmethod
    def demo(cls):
        data = flyingthings()
        print('train:', data.len['train'], ',val:', data.len['val'])
        split = 'train'
        for ind in range(data.len[split]):
            sample = data.sample(index=ind, split=split)
            im1, im2, flow = sample['im1'], sample['im2'], sample['flow']
            flow_im = tensor_tools.flow_to_image_dmax(flow)
            # tensor_tools.cv2_show_dict(im1=im1, im2=im2, flow_im=flow_im)
            print(sample['name'])
            tensor_tools.check_tensor_np(im1, name='im1')
            tensor_tools.check_tensor_np(im2, name='im2')
            tensor_tools.check_tensor_np(flow, name='flow')
            current_dir = os.path.split(os.path.realpath(__file__))[0]
            save_dir = os.path.join(current_dir, 'demo')
            file_tools.check_dir(save_dir)
            cv2.imwrite(os.path.join(current_dir, os.path.join(save_dir, '%s_im1_.png' % ind)), im1)
            cv2.imwrite(os.path.join(current_dir, os.path.join(save_dir, '%s_im2_.png' % ind)), im2)
            cv2.imwrite(os.path.join(current_dir, os.path.join(save_dir, '%s_flow_.png' % ind)), flow_im)
            if ind > 5:
                break

