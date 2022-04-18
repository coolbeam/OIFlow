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


class Sintel(tools.abs_database):
    def __init__(self, only_image=False, data_pass='clean', **kwargs):
        super(Sintel, self).__init__()
        self.root = '/data/Optical_Flow_all/datasets/MPI-Sintel-complete'
        current_dir = os.path.split(os.path.realpath(__file__))[0]
        self.data_pass = data_pass  # clean, final, both
        self.only_image = only_image
        # === dir path
        train_dir = os.path.join(self.root, 'training')
        test_dir = os.path.join(self.root, 'test')
        # === train dir path
        flow_dir = os.path.join(train_dir, 'flow')
        invalid_dir = os.path.join(train_dir, 'invalid')
        occ_dir = os.path.join(train_dir, 'occlusions')

        # training
        image_clean_dir = os.path.join(train_dir, 'clean')
        image_final_dir = os.path.join(train_dir, 'final')
        for scene in os.listdir(image_clean_dir):
            image_list_clean = sorted(glob(os.path.join(image_clean_dir, scene, '*.png')))
            image_list_final = sorted(glob(os.path.join(image_final_dir, scene, '*.png')))

            for i in range(len(image_list_clean) - 1):
                img_name = os.path.split(image_list_clean[i])[1]
                flow_name = img_name.replace('.png', '.flo')
                name = scene + '/' + img_name
                flow_path = os.path.join(flow_dir, scene, flow_name)
                invalid_path = os.path.join(invalid_dir, scene, img_name)
                occ_path = os.path.join(occ_dir, scene, img_name)
                image_clean = {'img1_path': image_list_clean[i], 'img2_path': image_list_clean[i + 1]}
                image_final = {'img1_path': image_list_final[i], 'img2_path': image_list_final[i + 1]}
                sample = {'flow_path': flow_path, 'name': name, 'clean': image_clean, 'final': image_final, 'invalid': invalid_path, 'occ': occ_path}
                self.data_ls['train'].append(sample)
        # test
        image_root_clean = os.path.join(test_dir, 'clean')
        image_root_final = os.path.join(test_dir, 'final')
        for scene in os.listdir(image_root_clean):
            image_list_clean = sorted(glob(os.path.join(image_root_clean, scene, '*.png')))
            image_list_final = sorted(glob(os.path.join(image_root_final, scene, '*.png')))
            for i in range(len(image_list_clean) - 1):
                name = scene + '/' + os.path.split(image_list_final[i])[1]
                clean_sample = {'img1_path': image_list_clean[i], 'img2_path': image_list_clean[i + 1]}
                final_sample = {'img1_path': image_list_final[i], 'img2_path': image_list_final[i + 1]}
                self.data_ls['test'].append({'clean': clean_sample, 'final': final_sample, 'name': name})
        self.data_ls['val'] = self.data_ls['train']
        self._init_len()

    # return {'im1','im2', 'flow', 'valid', 'name':, 'occ_mask', maybe with: {'final_im1','final_im2'}, occ_mask: occlusion pixels=1, valid: valid pixels=1
    def sample(self, index, split):
        sample = self.data_ls[split][index % self.len[split]]
        res = {}

        if self.data_pass in ['clean', 'final']:
            img1 = frame_utils.read_gen(sample[self.data_pass]['img1_path'])
            img2 = frame_utils.read_gen(sample[self.data_pass]['img2_path'])
        else:
            img1 = frame_utils.read_gen(sample['clean']['img1_path'])
            img2 = frame_utils.read_gen(sample['clean']['img2_path'])

            img1_ = frame_utils.read_gen(sample['final']['img1_path'])
            img2_ = frame_utils.read_gen(sample['final']['img2_path'])
            img1_ = np.array(img1_).astype(np.uint8)
            img2_ = np.array(img2_).astype(np.uint8)
            # grayscale images
            if len(img1_.shape) == 2:
                img1_ = np.tile(img1_[..., None], (1, 1, 3))
                img2_ = np.tile(img2_[..., None], (1, 1, 3))
            else:
                img1_ = img1_[..., :3]
                img2_ = img2_[..., :3]
            res.update({'final_im1': img1_, 'final_im2': img2_})
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
        res.update({'im1': img1, 'im2': img2, 'name': sample['name']})
        if split == 'test' or self.only_image:
            return res
        flow = frame_utils.read_gen(sample['flow_path'])
        flow = np.array(flow).astype(np.float32)
        valid = frame_utils.read_gen(sample['invalid'], read_mask=True)
        valid[valid > 0] = 1  # 0-1 mask, this is the invalid mask
        valid = 1 - valid
        # print(sample['occ'])
        occ_mask = frame_utils.read_gen(sample['occ'], read_mask=True)
        occ_mask[occ_mask > 0] = 1  # 0-1 mask, occlusion pixels=1
        valid = np.expand_dims(valid, axis=2)
        occ_mask = np.expand_dims(occ_mask, axis=2)
        res.update({'flow': flow, 'valid': valid, 'occ_mask': occ_mask})
        return res

    @classmethod
    def demo(cls):
        current_dir = os.path.split(os.path.realpath(__file__))[0]
        data_pass = 'both'
        data = Sintel(data_pass=data_pass)
        print('train:', data.len['train'], ',val:', data.len['val'], 'test', data.len['test'])
        split = 'train'
        for ind in range(data.len[split]):
            sample = data.sample(index=ind, split=split)
            im1, im2 = sample['im1'], sample['im2']
            if split == 'train':
                flow = sample['flow']
                valid_mask = sample['valid']
                occ_mask = sample['occ_mask']
                tensor_tools.check_tensor_np(flow, name='flow')
                tensor_tools.check_tensor_np(valid_mask, name='valid_mask')
                tensor_tools.check_tensor_np(occ_mask, name='occ_mask')
            # tensor_tools.cv2_show_dict(im1=im1, im2=im2, flow_im=flow_im)
            print(sample['name'])
            tensor_tools.check_tensor_np(im1, name='im1')
            tensor_tools.check_tensor_np(im2, name='im2')
            if ind > 10:
                # cv2.imwrite(current_dir + '/im1.png', im1[:, :, ::-1])
                # cv2.imwrite(current_dir + '/im2.png', im2[:, :, ::-1])
                # cv2.imwrite(current_dir + '/valid_mask.png', valid_mask * 250)
                # cv2.imwrite(current_dir + '/occ_mask.png', occ_mask * 250)
                # cv2.imwrite(current_dir + '/flow.png', tensor_tools.flow_to_image(flow))
                # if sintel_pass == 'both':
                #     f_im1, f_im2 = sample['final_im1'], sample['final_im2']
                #     tensor_tools.check_tensor_np(f_im1, name='f_im1')
                #     tensor_tools.check_tensor_np(f_im2, name='f_im2')
                #     cv2.imwrite(current_dir + '/final_im1.png', f_im1[:, :, ::-1])
                #     cv2.imwrite(current_dir + '/final_im2.png', f_im2[:, :, ::-1])
                break


