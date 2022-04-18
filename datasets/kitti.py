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


class KITTI(tools.abs_database):
    def __init__(self, only_image=False, data_pass='2012', **kwargs):
        super(KITTI, self).__init__()
        self.__root = '/data/Optical_Flow_all/datasets/KITTI_data'
        current_dir = os.path.split(os.path.realpath(__file__))[0]
        self.data_pass = data_pass  # 2012, 2015, 2012mv,2015mv
        self.only_image = only_image
        # === dir path
        self.__k2012_dir = os.path.join(self.__root, 'data_stereo_flow')
        self.__k2015_dir = os.path.join(self.__root, 'data_scene_flow')
        self.__kmv2012_dir = os.path.join(self.__root, 'KITTI_data_mv', 'stereo_flow_2012', 'data_stereo_flow_multiview')
        self.__kmv2015_dir = os.path.join(self.__root, 'KITTI_data_mv', 'stereo_flow_2015', 'data_scene_flow_multiview')

        data_ls = self.__get_data_ls()
        self.data_ls = data_ls[self.data_pass]
        self._init_len()

    def __get_data_ls(self):
        k2012_data = self.__kitti_2012(self.__k2012_dir)
        k2012mv_data = self.__kitti_mv_data(self.__kmv2012_dir)
        k2015_data = self.__kitti_2015(self.__k2015_dir)
        k2015mv_data = self.__kitti_mv_data(self.__kmv2015_dir)
        data_ls = {'2012': k2012_data,
                   '2012mv': k2012mv_data,
                   '2015': k2015_data,
                   '2015mv': k2015mv_data,
                   }
        return data_ls

    @classmethod
    def __get_img_flow_path_pair(cls, im_dir, flow_occ_dir, flow_noc_dir):
        a = []
        image_files = os.listdir(im_dir)
        image_files.sort()
        flow_occ_files = os.listdir(flow_occ_dir)
        flow_occ_files.sort()
        flow_noc_files = os.listdir(flow_noc_dir)
        flow_noc_files.sort()
        assert len(image_files) % 2 == 0, 'expected pairs of images'
        assert len(flow_occ_files) == len(flow_noc_files), 'here goes wrong'
        assert len(flow_occ_files) == len(image_files) / 2, 'here goes wrong'
        for i in range(len(image_files) // 2):
            filenames_1 = os.path.join(im_dir, image_files[i * 2])
            filenames_2 = os.path.join(im_dir, image_files[i * 2 + 1])
            filenames_gt_occ = os.path.join(flow_occ_dir, flow_occ_files[i])
            filenames_gt_noc = os.path.join(flow_noc_dir, flow_noc_files[i])
            name = os.path.split(filenames_1)[1]
            # print('occ', flow_occ_files[i], 'noc', flow_noc_files[i], 'im1', image_files[i * 2], 'im2', image_files[i * 2 + 1])
            a.append({'flow_occ': filenames_gt_occ, 'flow_noc': filenames_gt_noc, 'img1_path': filenames_1, 'img2_path': filenames_2, 'name': name})
        return a

    @classmethod
    def __get_img_path_dir(cls, im_dir):
        a = []
        image_files = os.listdir(im_dir)
        image_files.sort()
        assert len(image_files) % 2 == 0, 'expected pairs of images'
        for i in range(len(image_files) // 2):
            filenames_1 = os.path.join(im_dir, image_files[i * 2])
            filenames_2 = os.path.join(im_dir, image_files[i * 2 + 1])
            name = os.path.split(filenames_1)[1]
            a.append({'img1_path': filenames_1, 'img2_path': filenames_2, 'name': name})
        return a

    @classmethod
    def __kitti_2012(cls, data_dir):
        image_dir = os.path.join(data_dir, 'training', 'colored_0')
        flow_dir_occ = os.path.join(data_dir, 'training', 'flow_occ')
        flow_dir_noc = os.path.join(data_dir, 'training', 'flow_noc')
        train_ls = cls.__get_img_flow_path_pair(im_dir=image_dir, flow_occ_dir=flow_dir_occ, flow_noc_dir=flow_dir_noc)

        test_im_dir = os.path.join(data_dir, 'testing', 'colored_0')
        test_ls = cls.__get_img_path_dir(im_dir=test_im_dir)
        return {'train': train_ls, 'val': train_ls, 'test': test_ls}

    @classmethod
    def __kitti_2015(cls, data_dir):
        image_dir = os.path.join(data_dir, 'training', 'image_2')
        flow_dir_occ = os.path.join(data_dir, 'training', 'flow_occ')
        flow_dir_noc = os.path.join(data_dir, 'training', 'flow_noc')
        train_ls = cls.__get_img_flow_path_pair(im_dir=image_dir, flow_occ_dir=flow_dir_occ, flow_noc_dir=flow_dir_noc)

        test_im_dir = os.path.join(data_dir, 'testing', 'image_2')
        test_ls = cls.__get_img_path_dir(im_dir=test_im_dir)
        return {'train': train_ls, 'val': train_ls, 'test': test_ls}

    @classmethod
    def __kitti_mv_data(cls, data_dir):
        def read_mv_data(d_path):
            sample_ls = []
            for sub_dir in ['testing', 'training']:
                img_dir = os.path.join(d_path, sub_dir, 'image_2')
                file_ls = os.listdir(img_dir)
                file_ls.sort()
                for ind in range(len(file_ls) - 1):
                    name = file_ls[ind]
                    nex_name = file_ls[ind + 1]
                    id_ = int(name[-6:-4])
                    id_nex = int(nex_name[-6:-4])
                    if id_ != id_nex - 1 or 12 >= id_ >= 9 or 12 >= id_nex >= 9:
                        pass
                    else:
                        file_path = os.path.join(img_dir, name)
                        file_path_nex = os.path.join(img_dir, nex_name)
                        name_ = os.path.split(file_path)[1]
                        sample_ls.append({'img1_path': file_path, 'img2_path': file_path_nex, 'name': name_})

            return sample_ls

        data_ls = read_mv_data(data_dir)
        return {'train': data_ls, 'val': [], 'test': []}

    @classmethod
    def __readimg(cls, img_path):
        img1 = frame_utils.read_gen(img_path)
        img1 = np.array(img1).astype(np.uint8)
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
        return img1

    def __read_flow(self, flow_path):
        gtflow_all, valid_mask = frame_utils.read_png_flow(flow_path)
        if len(valid_mask.shape) == 2:
            valid_mask = np.expand_dims(valid_mask, axis=2)
        return gtflow_all, valid_mask

    # return {'im1','im2', 'flow', 'valid', 'name':, 'occ_mask', occ_mask: occlusion pixels=1, valid: valid pixels=1
    def sample(self, index, split):
        # {'flow_occ': flow_gt_occ, 'flow_noc': flow_gt_noc, 'img1_path': im1_path, 'img2_path': im2_path,'name':name}
        sample = self.data_ls[split][index % self.len[split]]
        res = {}
        # ==== read image
        img1 = self.__readimg(sample['img1_path'])
        img2 = self.__readimg(sample['img2_path'])
        res.update({'im1': img1, 'im2': img2, 'name': sample['name']})
        if split == 'test' or self.only_image or self.data_pass in ['2012mv', '2015mv']:
            return res
        # ==== read flow and valid mask
        gtflow_all, valid_mask = self.__read_flow(sample['flow_occ'])
        gtflow_noc, noc_mask = self.__read_flow(sample['flow_noc'])  # noc_mask: non-occlusion pixels =1 (in valid regions)
        occ_mask = valid_mask - noc_mask  # occlusion pixels=1
        res.update({'flow': gtflow_all, 'valid': valid_mask, 'occ_mask': occ_mask})
        return res

    @classmethod
    def demo(cls):
        current_dir = os.path.split(os.path.realpath(__file__))[0] + '/demo'
        file_tools.check_dir(current_dir)
        data_pass = '2015mv'
        data = KITTI(data_pass=data_pass)
        print('train:', data.len['train'], ',val:', data.len['val'], 'test', data.len['test'])
        split = 'train'
        for ind in range(data.len[split]):
            sample = data.sample(index=ind, split=split)
            im1, im2 = sample['im1'], sample['im2']
            print(sample['name'])
            if split != 'test' and data_pass not in ['2012mv', '2015mv']:
                tensor_tools.check_tensor_np(sample['flow'], name='flow')
                tensor_tools.check_tensor_np(sample['valid'], name='valid_mask')
                tensor_tools.check_tensor_np(sample['occ_mask'], name='occ_mask')
            # tensor_tools.cv2_show_dict(im1=im1, im2=im2, flow_im=flow_im)
            tensor_tools.check_tensor_np(im1, name='im1')
            tensor_tools.check_tensor_np(im2, name='im2')
            if ind > 10:
                cv2.imwrite(current_dir + '/im1.png', im1[:, :, ::-1])
                cv2.imwrite(current_dir + '/im2.png', im2[:, :, ::-1])
                if split != 'test' and data_pass not in ['2012mv', '2015mv']:
                    cv2.imwrite(current_dir + '/valid_mask.png', sample['valid'] * 250)
                    cv2.imwrite(current_dir + '/occ_mask.png', sample['occ_mask'] * 250)
                    cv2.imwrite(current_dir + '/flow.png', tensor_tools.flow_to_image(sample['flow']))
                break


