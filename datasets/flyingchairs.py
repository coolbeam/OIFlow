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


class flyingchairs(tools.abs_database):
    def __init__(self, only_image=False, **kwargs):
        super(flyingchairs, self).__init__()
        self.only_image = only_image
        self.root = '/data/Optical_Flow_all/datasets/FlyingChairs/FlyingChairs_release/data'
        images = sorted(glob(os.path.join(self.root, '*.ppm')))
        flows = sorted(glob(os.path.join(self.root, '*.flo')))
        assert (len(images) // 2 == len(flows))
        current_dir = os.path.split(os.path.realpath(__file__))[0]
        split_list = np.loadtxt(os.path.join(current_dir, 'files', 'chairs_split.txt'), dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            sample = {'flow_path': flows[i], 'img1_path': images[2 * i], 'img2_path': images[2 * i + 1], }
            if xid == 1:
                self.data_ls['train'].append(sample)
            elif xid == 2:
                self.data_ls['val'].append(sample)
            else:
                raise ValueError('wrong xid')
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
        data = flyingchairs()
        print('train:', data.len['train'], ',val:', data.len['val'])
        split = 'train'
        for ind in range(data.len[split]):
            sample = data.sample(index=ind, split=split)
            im1, im2, flow = sample['im1'], sample['im2'], sample['flow']
            flow_im = tensor_tools.flow_to_image(flow)
            # tensor_tools.cv2_show_dict(im1=im1, im2=im2, flow_im=flow_im)
            print(sample['name'])
            tensor_tools.check_tensor_np(im1, name='im1')
            tensor_tools.check_tensor_np(im2, name='im2')
            tensor_tools.check_tensor_np(flow, name='flow')
            if ind > 2:
                # cv2.imwrite('./im1.png', im1)
                # cv2.imwrite('./im2.png', im2)
                # cv2.imwrite('./flow.png', tensor_tools.flow_to_image(flow))
                break
