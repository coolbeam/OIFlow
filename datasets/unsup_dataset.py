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
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ColorJitter
from datasets.flyingchairs import flyingchairs
from datasets.sintel import Sintel
from datasets.kitti import KITTI

database_dict = {
    'flyingchairs': {'base': flyingchairs,  'sparse': False, 'img_size': (384, 512)},  # crop (320, 320)
    'sintel': {'base': Sintel,  'sparse': False, 'img_size': (436, 1024)},  # crop (320, 768)
    'KITTI': {'base': KITTI, 'sparse': True, 'img_size': (370, 1226)},  # crop (256, 832)
}


class Augmentor():
    class UnsupAugmentor(tools.abstract_config):
        def __init__(self, **kwargs):
            self.aug_color_prob = 0.9
            self.aug_color_asymmetric_prob = 0.2
            self.aug_eraser_prob = 0.5

            self.update(kwargs)
            # photometric augmentation params
            self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)

        def color_transform(self, img1, img2):
            """ Photometric augmentation """
            # no color aug
            if np.random.rand() > self.aug_color_prob:
                return img1, img2

            # asymmetric
            if np.random.rand() < self.aug_color_asymmetric_prob:
                # print('asymmetric aug')
                img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
                img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

            # symmetric
            else:
                # print('symmetric aug')
                image_stack = np.concatenate([img1, img2], axis=0)
                image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
                img1, img2 = np.split(image_stack, 2, axis=0)

            return img1, img2

        def eraser_transform(self, img1, img2, bounds=(50, 100)):
            """ Occlusion augmentation """
            ht, wd = img1.shape[:2]
            if np.random.rand() < self.aug_eraser_prob:
                # print('eraser aug')
                mean_color = np.mean(img2.reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 3)):
                    x0 = np.random.randint(0, wd)
                    y0 = np.random.randint(0, ht)
                    dx = np.random.randint(bounds[0], bounds[1])
                    dy = np.random.randint(bounds[0], bounds[1])
                    # print('erase h=%s, w=%s: [%s:%s, %s:%s,:]' % (ht, wd, y0, y0 + dy, x0, x0 + dx))
                    img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

            return img1, img2

        def __call__(self, img1, img2):
            # print('\n======')
            img1, img2 = self.color_transform(img1, img2)
            img1, img2 = self.eraser_transform(img1, img2)
            return img1, img2

        @classmethod
        def demo(cls):
            im1_p = r'C:\Users\28748\Documents\x2go_share\sintel\frame_0001.png'
            im2_p = r'C:\Users\28748\Documents\x2go_share\sintel\frame_0002.png'
            im1 = cv2.imread(im1_p)
            im2 = cv2.imread(im2_p)
            auger = Augmentor.UnsupAugmentor()
            for i in range(20):
                im1_a, im2_a = auger(img1=im1, img2=im2)
                tensor_tools.cv2_show_dict(im1=im1, im2=im2, im1_a=im1_a, im2_a=im2_a)


class UnFlowDataset(Dataset):
    class Config(tools.abstract_config):
        def __init__(self, **kwargs):
            self.data_name = 'flyingchairs'
            self.data_pass = 'clean'
            self.datatype = 'base'  # maybe use nori to fetch dataset. Nori is a dataset format mainly used in brain++ platform in Megvii. You should use 'base' here if you can not use nori.
            self.split = 'train'  # val, test
            # read_all_in_memory = False:not implemented uint8 (100, 8, 436, 1024)need 340M, float32 (100, 2, 436, 1024)need 340M. Thus 2000 pair need 14G memory.

            # === augmentation params
            self.aug_sintel_final_prob = 0
            self.aug_switch_prob = 0.5
            self.aug_crop_size = (320, 480)  # h,w
            self.aug_crop_rho = 8
            self.aug_horizontal_prob = 0.5
            self.aug_vertical_prob = 0.1
            # photo metric and occlusion aug
            self.aug_color_prob = 0.9
            self.aug_color_asymmetric_prob = 0.2
            self.aug_eraser_prob = 0.5

            self.update(kwargs)
            # need to fetch clean and final img simultaneously
            if self.aug_sintel_final_prob > 0:
                assert self.data_name == 'sintel'
                self.data_pass = 'both'  # not clean or final, because we want to read clean and final img simultaneously

        def __call__(self):
            return UnFlowDataset(self)

        def fetch_prefether(self, batch_size):
            dataset = self()
            train_loader = tools.data_prefetcher_dict(dataset, gpu_keys=('im1', 'im2', 'flow', 'valid'),
                                                      batch_size=batch_size, pin_memory=False, shuffle=True,
                                                      num_workers=4, drop_last=True)
            return train_loader

    def __init__(self, conf: Config):
        self.conf = conf
        data_base_func = database_dict[self.conf.data_name][self.conf.datatype]
        self.data_base = data_base_func(only_image=True, **self.conf.to_dict)  # get data base
        self.auger = Augmentor.UnsupAugmentor(**self.conf.to_dict)
        self.len = self.data_base.len[self.conf.split]
        self.len_rmul = 1

    def _random_flip(self, *args):
        # given h,w,c image
        def temp_w(a):
            b = np.flip(a, 1)
            return b

        def temp_h(a):
            b = np.flip(a, 0)
            return b

        res = args
        if np.random.rand() < self.conf.aug_horizontal_prob:
            res = [temp_w(i) for i in res]
        if np.random.rand() < self.conf.aug_vertical_prob:
            res = [temp_h(i) for i in res]
        return res

    def _random_crop(self, *args):
        height, width = args[0].shape[:2]
        # print(height, width)
        patch_size_h, patch_size_w = self.conf.aug_crop_size
        x = np.random.randint(self.conf.aug_crop_rho, width - self.conf.aug_crop_rho - patch_size_w)
        # print(self.rho, height - self.rho - patch_size_h)
        y = np.random.randint(self.conf.aug_crop_rho, height - self.conf.aug_crop_rho - patch_size_h)
        start = np.array([x, y])
        start = np.expand_dims(np.expand_dims(start, 0), 0)
        data_ls = []
        for i in args:
            i_patch = i[y: y + patch_size_h, x: x + patch_size_w, :]
            data_ls.append(i_patch)
        data_ls.append(start)
        return data_ls

    def _random_switch(self, *args):
        res = []
        if np.random.rand() < self.conf.aug_switch_prob:
            for data in args:
                im1, im2 = data
                res.append((im2, im1))
        else:
            for data in args:
                res.append(data)
        return res

    def _np_hwc_to_tensor(self, *args):
        def func(a):
            # this is important, array should be contiguous (img after flip is not contiguous)
            a = np.ascontiguousarray(a)
            # tensor_tools.check_tensor_np(a, 'a')
            b = torch.from_numpy(a).permute(2, 0, 1).float()
            return b

        return [func(i) for i in args]

    def __getitem__(self, index):
        sample = self.data_base.sample(index=index, split=self.conf.split)
        img1, img2 = sample['im1'], sample['im2']  # h,w,c
        name = sample['name']
        # use final img as color aug img
        if np.random.rand() < self.conf.aug_sintel_final_prob:
            img1_aug, img2_aug = sample['final_im1'], sample['final_im2']
            (img1, img2), (img1_aug, img2_aug) = self._random_switch((img1, img2), (img1_aug, img2_aug))
        else:
            img1_aug, img2_aug = img1.copy(), img2.copy()
            (img1, img2), (img1_aug, img2_aug) = self._random_switch((img1, img2), (img1_aug, img2_aug))
        img1, img2, img1_aug, img2_aug = self._random_flip(img1, img2, img1_aug, img2_aug)

        # do photometric aug and occ aug
        if self.auger is not None:
            img1_aug, img2_aug = self.auger(img1_aug, img2_aug)
        # h,w,c
        im1_crop, im2_crop, img1_aug_crop, img2_aug_crop, start = self._random_crop(img1, img2, img1_aug, img2_aug)
        # transfer to tensor
        img1, img2, im1_crop, im2_crop, img1_aug_crop, img2_aug_crop, start = \
            self._np_hwc_to_tensor(img1, img2, im1_crop, im2_crop, img1_aug_crop, img2_aug_crop, start)

        res = {'im1': img1, 'im2': img2, 'im1_crop': im1_crop, 'im2_crop': im2_crop, 'name': name,
               'im1_crop_aug': img1_aug_crop, 'im2_crop_aug': img2_aug_crop, 'crop_start': start}  # chw
        return res

    def __rmul__(self, v):
        self.len_rmul = v * self.len_rmul
        return self

    def __mul__(self, v):
        self.len_rmul = v * self.len_rmul
        return self

    def __len__(self):
        return self.len * self.len_rmul

    @classmethod
    def demo(cls, if_save_img=False):
        def fetch_data(data, *args):
            res = []
            for i in args:
                res.append(data[i])
            return res

        def process(*args):
            def temp(a):
                b = a.numpy()
                b = np.squeeze(b)
                b = np.transpose(b, (1, 2, 0))
                b = tensor_tools.im_norm(b)
                b = b.astype(np.uint8)
                return b

            return [temp(i) for i in args]

        def save_data_samples(dataset, data_name):
            current_dir = os.path.split(os.path.realpath(__file__))[0]
            save_dir = os.path.join(current_dir, data_name)
            file_tools.check_dir(save_dir)
            print('===')
            print(len(dataset))
            print('===')
            for i in range(len(dataset)):
                sample = dataset[i * 100 + 11]  # = data.__getitem__(i)
                im1, im2, im1_crop, im2_crop, im1_crop_aug, im2_crop_aug = fetch_data(sample, 'im1', 'im2', 'im1_crop', 'im2_crop', 'im1_crop_aug', 'im2_crop_aug')
                name = sample['name']
                name = name.replace('/', '_')
                crop_start = sample['crop_start']
                print(data_name + ': ', name)
                im1, im2, im1_crop, im2_crop, im1_crop_aug, im2_crop_aug = process(*fetch_data(sample, 'im1', 'im2', 'im1_crop', 'im2_crop', 'im1_crop_aug', 'im2_crop_aug'))
                tensor_tools.check_tensor_np(im1, 'im1')
                tensor_tools.check_tensor_np(im2, 'im2')
                tensor_tools.check_tensor_np(im1_crop, 'im1_crop')
                tensor_tools.check_tensor_np(im2_crop, 'im2_crop')
                tensor_tools.check_tensor_np(im1_crop_aug, 'im1_crop_aug')
                tensor_tools.check_tensor_np(im2_crop_aug, 'im2_crop_aug')
                cv2.imwrite(os.path.join(save_dir, '%s_%s.png' % (name, 'im1')), im1)
                cv2.imwrite(os.path.join(save_dir, '%s_%s.png' % (name, 'im2')), im2)
                cv2.imwrite(os.path.join(save_dir, '%s_%s.png' % (name, 'im1_crop')), im1_crop)
                cv2.imwrite(os.path.join(save_dir, '%s_%s.png' % (name, 'im2_crop')), im2_crop)
                cv2.imwrite(os.path.join(save_dir, '%s_%s.png' % (name, 'im1_crop_aug')), im1_crop_aug)
                cv2.imwrite(os.path.join(save_dir, '%s_%s.png' % (name, 'im2_crop_aug')), im2_crop_aug)
                if i > 5:
                    break

        def check_data_sample(dataset, data_name):
            print('===')
            print(len(dataset))
            print('===')
            for i in range(len(dataset)):
                sample = dataset[i]  # = data.__getitem__(i)
                im1, im2 = sample['im1'], sample['im2']
                im1_crop, im2_crop = sample['im1_crop'], sample['im2_crop']
                im1_crop_aug, im2_crop_aug = sample['im1_crop_aug'], sample['im2_crop_aug']
                name = sample['name']
                crop_start = sample['crop_start']
                print(data_name + ': ', name)
                tensor_tools.check_tensor(im1, 'im1')
                tensor_tools.check_tensor(im2, data_name + 'im2')
                tensor_tools.check_tensor(im1_crop, 'im1_crop')
                tensor_tools.check_tensor(im2_crop, 'im2_crop')
                tensor_tools.check_tensor(im1_crop_aug, 'im1_crop_aug')
                tensor_tools.check_tensor(im2_crop_aug, 'im2_crop_aug')
                tensor_tools.check_tensor(crop_start, 'crop_start')
                if i > 5:
                    break

        d_name = 'KITTI'
        d_pass = '2012mv'
        conf_dict = {'data_name': d_name, 'data_pass': d_pass,
                     # aug param
                     'aug_sintel_final_prob': 0,
                     'aug_crop_size': (256, 832),  # chairs:(320, 480), sintel:(320, 768), KITTI: (256, 832)
                     'aug_color_prob': 0.9,
                     'aug_color_asymmetric_prob': 0.2,
                     'aug_eraser_prob': 0.5
                     }
        d_conf = UnFlowDataset.Config(**conf_dict)
        data = d_conf() * 10
        # check_data_sample(dataset=data, data_name=d_name + '_' + d_pass)
        check_data_sample(dataset=data, data_name=d_name + '_' + d_pass)

