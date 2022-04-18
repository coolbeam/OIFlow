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
from PIL import Image
from torchvision.transforms import ColorJitter
from datasets.flyingchairs import flyingchairs
from datasets.flyingthings import flyingthings
from datasets.sintel import Sintel
from datasets.kitti import KITTI

database_dict = {
    'flyingchairs': {'base': flyingchairs, 'sparse': False},
    'flyingthings': {'base': flyingthings, 'sparse': False},
    'sintel': {'base': Sintel, 'sparse': False},
    'KITTI': {'base': KITTI, 'sparse': True},
}


class Augmentor():
    class FlowAugmentor:
        def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):

            # spatial augmentation params
            self.crop_size = crop_size
            self.min_scale = min_scale
            self.max_scale = max_scale
            self.spatial_aug_prob = 0.8
            self.stretch_prob = 0.8
            self.max_stretch = 0.2

            # flip augmentation params
            self.do_flip = do_flip
            self.h_flip_prob = 0.5
            self.v_flip_prob = 0.1

            # photometric augmentation params
            self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
            self.asymmetric_color_aug_prob = 0.2
            self.eraser_aug_prob = 0.5

        def color_transform(self, img1, img2):
            """ Photometric augmentation """

            # asymmetric
            if np.random.rand() < self.asymmetric_color_aug_prob:
                img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
                img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

            # symmetric
            else:
                image_stack = np.concatenate([img1, img2], axis=0)
                image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
                img1, img2 = np.split(image_stack, 2, axis=0)

            return img1, img2

        def eraser_transform(self, img1, img2, bounds=(50, 100)):
            """ Occlusion augmentation """

            ht, wd = img1.shape[:2]
            if np.random.rand() < self.eraser_aug_prob:
                mean_color = np.mean(img2.reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 3)):
                    x0 = np.random.randint(0, wd)
                    y0 = np.random.randint(0, ht)
                    dx = np.random.randint(bounds[0], bounds[1])
                    dy = np.random.randint(bounds[0], bounds[1])
                    img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

            return img1, img2

        def spatial_transform(self, img1, img2, flow, valid=None):
            # randomly sample scale
            ht, wd = img1.shape[:2]
            min_scale = np.maximum(
                (self.crop_size[0] + 8) / float(ht),
                (self.crop_size[1] + 8) / float(wd))

            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = scale
            scale_y = scale
            if np.random.rand() < self.stretch_prob:
                scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_x = np.clip(scale_x, min_scale, None)
            scale_y = np.clip(scale_y, min_scale, None)

            if np.random.rand() < self.spatial_aug_prob:
                # rescale the images
                img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                flow = flow * [scale_x, scale_y]
            if self.do_flip:
                if np.random.rand() < self.h_flip_prob:  # h-flip
                    img1 = img1[:, ::-1]
                    img2 = img2[:, ::-1]
                    flow = flow[:, ::-1] * [-1.0, 1.0]

                if np.random.rand() < self.v_flip_prob:  # v-flip
                    img1 = img1[::-1, :]
                    img2 = img2[::-1, :]
                    flow = flow[::-1, :] * [1.0, -1.0]

            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

            return img1, img2, flow, valid

        def __call__(self, img1, img2, flow, valid=None):
            img1, img2 = self.color_transform(img1, img2)
            img1, img2 = self.eraser_transform(img1, img2)
            img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid=valid)

            img1 = np.ascontiguousarray(img1)
            img2 = np.ascontiguousarray(img2)
            flow = np.ascontiguousarray(flow)
            if valid is not None:
                valid = np.ascontiguousarray(valid)

            return img1, img2, flow, valid

    class SparseFlowAugmentor:
        def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
            # spatial augmentation params
            self.crop_size = crop_size
            self.min_scale = min_scale
            self.max_scale = max_scale
            self.spatial_aug_prob = 0.8
            self.stretch_prob = 0.8
            self.max_stretch = 0.2

            # flip augmentation params
            self.do_flip = do_flip
            self.h_flip_prob = 0.5
            self.v_flip_prob = 0.1

            # photometric augmentation params
            self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
            self.asymmetric_color_aug_prob = 0.2
            self.eraser_aug_prob = 0.5

        def color_transform(self, img1, img2):
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)
            return img1, img2

        def eraser_transform(self, img1, img2):
            ht, wd = img1.shape[:2]
            if np.random.rand() < self.eraser_aug_prob:
                mean_color = np.mean(img2.reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 3)):
                    x0 = np.random.randint(0, wd)
                    y0 = np.random.randint(0, ht)
                    dx = np.random.randint(50, 100)
                    dy = np.random.randint(50, 100)
                    img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

            return img1, img2

        def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
            ht, wd = flow.shape[:2]
            coords = np.meshgrid(np.arange(wd), np.arange(ht))
            coords = np.stack(coords, axis=-1)

            coords = coords.reshape(-1, 2).astype(np.float32)
            flow = flow.reshape(-1, 2).astype(np.float32)
            valid = valid.reshape(-1).astype(np.float32)

            coords0 = coords[valid >= 1]
            flow0 = flow[valid >= 1]

            ht1 = int(round(ht * fy))
            wd1 = int(round(wd * fx))

            coords1 = coords0 * [fx, fy]
            flow1 = flow0 * [fx, fy]

            xx = np.round(coords1[:, 0]).astype(np.int32)
            yy = np.round(coords1[:, 1]).astype(np.int32)

            v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
            xx = xx[v]
            yy = yy[v]
            flow1 = flow1[v]

            flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
            valid_img = np.zeros([ht1, wd1], dtype=np.int32)

            flow_img[yy, xx] = flow1
            valid_img[yy, xx] = 1

            return flow_img, valid_img

        def spatial_transform(self, img1, img2, flow, valid):
            # randomly sample scale

            ht, wd = img1.shape[:2]
            min_scale = np.maximum(
                (self.crop_size[0] + 1) / float(ht),
                (self.crop_size[1] + 1) / float(wd))

            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = np.clip(scale, min_scale, None)
            scale_y = np.clip(scale, min_scale, None)

            if np.random.rand() < self.spatial_aug_prob:
                # rescale the images
                img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

            if self.do_flip:
                if np.random.rand() < 0.5:  # h-flip
                    img1 = img1[:, ::-1]
                    img2 = img2[:, ::-1]
                    flow = flow[:, ::-1] * [-1.0, 1.0]
                    valid = valid[:, ::-1]

            margin_y = 20
            margin_x = 50

            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
            x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

            y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
            x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            return img1, img2, flow, valid

        def __call__(self, img1, img2, flow, valid):
            img1, img2 = self.color_transform(img1, img2)
            img1, img2 = self.eraser_transform(img1, img2)
            img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

            img1 = np.ascontiguousarray(img1)
            img2 = np.ascontiguousarray(img2)
            flow = np.ascontiguousarray(flow)
            valid = np.ascontiguousarray(valid)

            return img1, img2, flow, valid


class FlowDataset(Dataset):
    class Config(tools.abstract_config):
        def __init__(self, **kwargs):
            self.data_name = 'flyingchairs'
            self.data_pass = 'clean'
            self.aug_params = None
            self.datatype = 'base'  # maybe use nori to fetch dataset. Nori is a dataset format mainly used in brain++ platform in Megvii. You should use 'base' here if you can not use nori.
            self.split = 'train'  # val, test
            self.init_seed = False  # no use but i don't know its usage

            self.sparse = False  # will be initialized in __call__
            self.update(kwargs)

        def __call__(self):
            self.sparse = database_dict[self.data_name]['sparse']
            return FlowDataset(self)

        def fetch_prefether(self, batch_size):
            dataset = self()
            train_loader = tools.data_prefetcher_dict(dataset, gpu_keys=('im1', 'im2', 'flow', 'valid'),
                                                      batch_size=batch_size, pin_memory=False, shuffle=True,
                                                      num_workers=4, drop_last=True)
            return train_loader

    def __init__(self, conf: Config):
        self.conf = conf
        self.augmentor = None
        if self.conf.aug_params is not None:
            if self.conf.sparse:
                self.augmentor = Augmentor.SparseFlowAugmentor(**self.conf.aug_params)
            else:
                self.augmentor = Augmentor.FlowAugmentor(**self.conf.aug_params)
        data_base_func = database_dict[self.conf.data_name][self.conf.datatype]
        self.data_base = data_base_func(**self.conf.to_dict)  # get data base
        self.len = self.data_base.len[self.conf.split]
        self.len_rmul = 1

    def __getitem__(self, index):
        if self.conf.split == 'test':
            sample = self.data_base.sample(index=index, split=self.conf.split)
            img1, img2 = sample['im1'], sample['im2']  # h,w,c
            name = sample['name']
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return {'im1': img1, 'im2': img2, 'name': name}

        # if not self.conf.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.conf.init_seed = True

        sample = self.data_base.sample(index=index, split=self.conf.split)
        img1, img2, flow = sample['im1'], sample['im2'], sample['flow']  # h,w,c
        valid = sample['valid']
        name = sample['name']

        if self.augmentor is not None:
            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        if len(valid.shape) == 2:
            valid = torch.unsqueeze(valid, 0)  # 1,h,w
        elif valid.shape[2] == 1:
            valid = valid.permute(2, 0, 1)
        res = {'im1': img1, 'im2': img2, 'flow': flow, 'valid': valid.float(), 'name': name}
        if 'occ_mask' in sample.keys() and self.conf.split != 'train':
            occ_mask = sample['occ_mask']
            if len(occ_mask.shape) == 2:
                occ_mask = np.expand_dims(occ_mask, 0)  # 1,h,w
            elif occ_mask.shape[2] == 1:
                occ_mask = np.transpose(occ_mask, (2, 0, 1))
            occ_mask = torch.from_numpy(occ_mask).float()
            res['occ_mask'] = occ_mask
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
    def demo(cls):
        def process(*args):
            def temp(a):
                b = a.numpy()
                b = np.squeeze(b)
                b = np.transpose(b, (1, 2, 0))
                b = tensor_tools.im_norm(b)
                b = b.astype(np.uint8)
                return b

            return [temp(i) for i in args]

        aug_params = {'crop_size': (368, 496), 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        d_conf = FlowDataset.Config(data_name='sintel', split='train', aug_params=aug_params, sintel_pass='clean')
        data = d_conf()
        d_conf2 = FlowDataset.Config(data_name='sintel', split='train', aug_params=aug_params, sintel_pass='final')
        data2 = d_conf2() * 10
        new_data = data + data2

        print('===')
        print(len(data))
        print(len(data2))
        print(len(new_data))
        print('===')
        for i in range(len(new_data)):
            sample = data[i]  # = data.__getitem__(i)
            im1, im2 = sample['im1'], sample['im2']
            tensor_tools.check_tensor(im1, 'im1')
            tensor_tools.check_tensor(im2, 'im2')
            if i > 5:
                break

    @classmethod
    def demo_loader(cls):
        def process(*args):
            def temp(a):
                b = a.numpy()
                b = np.squeeze(b)
                b = np.transpose(b, (1, 2, 0))
                b = tensor_tools.im_norm(b)
                b = b.astype(np.uint8)
                return b

            return [temp(i) for i in args]

        aug_params = {'crop_size': (368, 496), 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        d_conf = FlowDataset.Config(data_name='flyingchairs', split='train', aug_params=aug_params)
        data = d_conf()
        print(len(data))
        print('===')

        train_loader = DataLoader(data, batch_size=4, pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
        for i, batch in enumerate(train_loader):
            sample = batch
            print(type(sample))
            print(sample.keys())
            im1, im2 = sample['im1'], sample['im2']
            tensor_tools.check_tensor(im1, 'im1')
            tensor_tools.check_tensor(im2, 'im2')
            tensor_tools.check_tensor(sample['flow'], 'flow')
            print('%s  \n' % sample['name'])
            if i > 3:
                break

    @classmethod
    def demo_prefetcher(cls):
        def process(*args):
            def temp(a):
                b = a.numpy()
                b = np.squeeze(b)
                b = np.transpose(b, (1, 2, 0))
                b = tensor_tools.im_norm(b)
                b = b.astype(np.uint8)
                return b

            return [temp(i) for i in args]

        aug_params = {'crop_size': (368, 496), 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        d_conf = FlowDataset.Config(data_name='sintel', split='train', aug_params=aug_params, data_pass='clean')
        data = d_conf()
        print(len(data))
        print('===')

        # train_loader = tools.data_prefetcher_dict(data, gpu_keys=('im1', 'flow'), batch_size=4, pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
        train_loader = tools.data_prefetcher_dict(data, gpu_keys=('im1', 'im2', 'flow', 'valid', 'occ_mask'),
                                                  batch_size=4, pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
        batch = train_loader.next()
        ind = 0
        while batch is not None:
            ind += 1
            sample = batch
            print(type(sample))
            print(sample.keys())
            im1, im2 = sample['im1'], sample['im2']
            tensor_tools.check_tensor(im1, 'im1')
            tensor_tools.check_tensor(im2, 'im2')
            tensor_tools.check_tensor(sample['flow'], 'flow')
            tensor_tools.check_tensor(sample['valid'], 'valid')
            tensor_tools.check_tensor(sample['occ_mask'], 'occ_mask')
            print('%s  \n' % sample['name'])
            batch = train_loader.next()
            if ind > 3:
                break
