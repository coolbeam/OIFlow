import os
from utils.tools import tools, tensor_tools, file_tools, frame_utils
import random
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from datasets.flow_dataset import FlowDataset, database_dict
import torch.nn.functional as F

eval_dict = {
    'flyingchairs': {'eval_metric': 'EPE', 'data_name': 'flyingchairs', 'eval_pad': True},
    'sintel_clean': {'eval_metric': 'EPEocc', 'data_name': 'sintel', 'data_pass': 'clean', 'eval_pad': False},
    'sintel_final': {'eval_metric': 'EPEocc', 'data_name': 'sintel', 'data_pass': 'final', 'eval_pad': False},
    'KITTI_2012': {'eval_metric': 'EPEocc', 'data_name': 'KITTI', 'data_pass': '2012', 'eval_pad': False},
    'KITTI_2015': {'eval_metric': 'EPEocc', 'data_name': 'KITTI', 'data_pass': '2015', 'eval_pad': False},
}


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', eval_pad_rate=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // eval_pad_rate) + 1) * eval_pad_rate - self.ht) % eval_pad_rate
        pad_wd = (((self.wd // eval_pad_rate) + 1) * eval_pad_rate - self.wd) % eval_pad_rate
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class FlowEvalBench(tools.abstract_config):
    def __init__(self, **kwargs):
        self.if_gpu = True
        self.eval_batch_size = 1
        self.eval_name = 'flyingchairs'
        self.eval_datatype = 'nori'  # or base
        self.if_print_process = False
        self.eval_save_results = False
        self.if_shuffle = True  # shuffle=False when doing evaluation

        self.update(kwargs)
        self.eval_metric = eval_dict[self.eval_name]['eval_metric']
        self.eval_pad = eval_dict[self.eval_name]['eval_pad']
        if self.eval_name in ['KITTI_2012', 'KITTI_2015']:
            self.eval_batch_size = 1
            print('=' * 10)
            print('change eval batch size =1 for %s' % self.eval_name)
            print('=' * 10)
        try:
            self.dataloader = self.fetch_dataloader()
        except:
            pass
        self.timer = tools.TimeClock()

    def fetch_dataloader(self):
        gpu_keys_ = {'EPE': ('im1', 'im2', 'flow', 'valid'),
                     'EPEocc': ('im1', 'im2', 'flow', 'valid', 'occ_mask'),
                     }
        d_conf = FlowDataset.Config(split='val', aug_params=None, datatype=self.eval_datatype, **eval_dict[self.eval_name])
        dataset = d_conf()
        dataloader = tools.data_prefetcher_dict(dataset, gpu_keys=gpu_keys_[self.eval_metric],
                                                batch_size=self.eval_batch_size, pin_memory=False, shuffle=self.if_shuffle, num_workers=4, drop_last=False)
        return dataloader

    def eval_only_epe(self, test_model: tools.abs_test_model):
        error_meter = tools.AverageMeter()
        self.timer.start()
        index = -1
        batch = self.dataloader.next()
        test_model.eval()
        with torch.no_grad():
            while batch is not None:
                index += 1
                im1, im2 = batch['im1'], batch['im2']
                gtflow = batch['flow']
                valid = batch['valid']
                # test
                if self.eval_pad:
                    padder = InputPadder(im1.shape)
                    im1, im2 = padder.pad(im1, im2)
                    predflow = test_model.eval_forward(im1=im1, im2=im2, gtflow=gtflow)
                    predflow = padder.unpad(predflow)
                else:
                    predflow = test_model.eval_forward(im1=im1, im2=im2, gtflow=gtflow)
                # error
                num = im1.shape[0]
                epe_error = self.flow_epe_error(predflow=predflow, gt_flow=gtflow, mask=valid)
                error_meter.update(val=epe_error.item(), num=num)
                if self.eval_save_results:
                    test_model.record_eval_score(epe_error.item())
                if self.if_print_process and index % 20 == 0:
                    print('%s, EPE = %.2f(avg=%.2f)' % (index, error_meter.val, error_meter.avg))
                batch = self.dataloader.next()
        if self.eval_save_results:
            test_model.save_record()
        self.timer.end()
        print('=' * 3 + ' eval time %.1f s ' % self.timer.get_during() + '=' * 3)
        print('%s, everage EPE = %.2f' % (self.eval_name, error_meter.avg))
        # info = {'%s_%s' % (self.eval_name, self.eval_metric): error_meter.avg}
        info = {'EPE': error_meter.avg, 'name': self.eval_name}
        return error_meter.avg, info

    def eval_only_epe_raft(self, test_model: tools.abs_test_model):
        # assert self.eval_batch_size == 1
        epe_list = []
        self.timer.start()
        index = -1
        batch = self.dataloader.next()
        test_model.eval()
        with torch.no_grad():
            while batch is not None:
                index += 1
                im1, im2 = batch['im1'], batch['im2']
                gtflow = batch['flow']
                valid = batch['valid']
                # test
                if self.eval_pad:
                    padder = InputPadder(im1.shape)
                    im1, im2 = padder.pad(im1, im2)
                    predflow = test_model.eval_forward(im1=im1, im2=im2, gtflow=gtflow)
                    predflow = padder.unpad(predflow)
                else:
                    predflow = test_model.eval_forward(im1=im1, im2=im2, gtflow=gtflow)
                # error
                epe = torch.sum((predflow - gtflow) ** 2, dim=1).sqrt()
                if epe.is_cuda:
                    epe = epe.cpu()
                epe_list.append(epe.view(-1).numpy())
                if self.eval_save_results:
                    test_model.record_eval_score(epe_list[-1])
                if self.if_print_process and index % 20 == 0:
                    print(index)
                batch = self.dataloader.next()
        if self.eval_save_results:
            test_model.save_record()
        epe = np.mean(np.concatenate(epe_list))
        self.timer.end()
        print('=' * 3 + ' eval time %.1f s ' % self.timer.get_during() + '=' * 3)
        print("Validation EPE: %f" % epe)
        info = {'EPE': epe, 'name': self.eval_name}
        return epe, info

    def eval_epe_occ(self, test_model: tools.abs_test_model):
        all_pep_meter = tools.AverageMeter()
        f1_rate_meter = tools.AverageMeter()
        occ_pep_meter = tools.AverageMeter()
        noc_pep_meter = tools.AverageMeter()

        self.timer.start()
        index = -1
        batch = self.dataloader.next()
        test_model.eval()
        with torch.no_grad():
            while batch is not None:
                index += 1
                im1, im2 = batch['im1'], batch['im2']
                gtflow = batch['flow']
                valid = batch['valid']
                occ_mask = batch['occ_mask']
                # test
                if self.eval_pad:
                    padder = InputPadder(im1.shape)
                    im1, im2 = padder.pad(im1, im2)
                    predflow = test_model.eval_forward(im1=im1, im2=im2, gtflow=gtflow, valid=valid, occ_mask=occ_mask)
                    predflow = padder.unpad(predflow)
                else:
                    predflow = test_model.eval_forward(im1=im1, im2=im2, gtflow=gtflow, valid=valid, occ_mask=occ_mask)  # give gtflow, valid and occ mask for saving results
                # tensor_tools.check_tensor(im1, '%s_im1' % index)
                # error
                num = im1.shape[0]
                epe_error = self.flow_epe_error(predflow=predflow, gt_flow=gtflow, mask=valid)

                f1_rate = self.flow_f1_pct(gt_flow=gtflow, predflow=predflow, mask=valid)
                noc_pep_error_ = self.flow_epe_error(predflow=predflow, gt_flow=gtflow, mask=valid * (1 - occ_mask))
                occ_pep_error_ = self.flow_epe_error(predflow=predflow, gt_flow=gtflow, mask=valid * occ_mask)

                all_pep_meter.update(val=epe_error.item(), num=num)
                f1_rate_meter.update(val=f1_rate.item(), num=num)
                occ_pep_meter.update(val=occ_pep_error_.item(), num=num)
                noc_pep_meter.update(val=noc_pep_error_.item(), num=num)
                if self.eval_save_results:
                    test_model.record_eval_score(epe_error.item())
                if self.if_print_process and index % 20 == 0:
                    print('%s, EPE = %.2f(avg=%.2f), NOC = %.2f(avg=%.2f), OCC = %.2f(avg=%.2f), F1 = %.2f(avg=%.2f)' % (index,
                                                                                                                         all_pep_meter.val, all_pep_meter.avg,
                                                                                                                         noc_pep_meter.val, noc_pep_meter.avg,
                                                                                                                         occ_pep_meter.val, occ_pep_meter.avg,
                                                                                                                         f1_rate_meter.val, f1_rate_meter.avg))
                batch = self.dataloader.next()
        if self.eval_save_results:
            test_model.save_record()
        self.timer.end()
        print('=' * 3 + ' eval time %.1f s ' % self.timer.get_during() + '=' * 3)
        print('%s, EPE = %.2f, NOC = %.2f, OCC = %.2f, F1 = %.2f' % (self.eval_name, all_pep_meter.avg, noc_pep_meter.avg, occ_pep_meter.avg, f1_rate_meter.avg))
        # res = {'%s_%s' % (self.eval_name, 'EPE'): all_pep_meter.avg, '%s_%s' % (self.eval_name, 'NOC'): noc_pep_meter.avg,
        #        '%s_%s' % (self.eval_name, 'OCC'): occ_pep_meter.avg, '%s_%s' % (self.eval_name, 'F1'): f1_rate_meter.avg}
        info = {'EPE': all_pep_meter.avg, 'name': self.eval_name, 'NOC': noc_pep_meter.avg, 'OCC': occ_pep_meter.avg, 'F1': f1_rate_meter.avg}
        return all_pep_meter.avg, info

    def __call__(self, test_model: tools.abs_test_model):
        func_list = {
            'EPE': self.eval_only_epe,
            'EPE_raft': self.eval_only_epe_raft,
            'EPEocc': self.eval_epe_occ,
        }
        func = func_list[self.eval_metric]
        return func(test_model)

    @classmethod
    def flow_epe_error(cls, predflow, gt_flow, mask=None):
        """
        Evaluates the average endpoint error between flow batches. torch batch is n c h w

        """

        def euclidean(t):
            return torch.sqrt(torch.sum(t ** 2, dim=(1,), keepdim=True))

        def batch_sum(a):
            b = torch.sum(a.float(), dim=2, keepdim=True)
            b = torch.sum(b, dim=3, keepdim=True)
            return b

        if mask is None:
            mask = torch.ones_like(predflow.narrow(1, 0, 1))
        if len(mask.shape) == 3:  # n,h,w
            mask = torch.unsqueeze(mask, 1)

        diff = euclidean(predflow - gt_flow) * mask  # n, 1, h, w
        mask_s = batch_sum(mask)
        diff_s = batch_sum(diff)
        error = diff_s / (mask_s + 1e-6)  # n,1,1,1
        return torch.mean(error)

    @classmethod
    def flow_f1_pct(cls, gt_flow, predflow, mask, threshold=3.0, relative=0.05):
        def euclidean(t):
            return torch.sqrt(torch.sum(t ** 2, dim=(1,), keepdim=True))

        def batch_sum(a):
            b = torch.sum(a.float(), dim=2, keepdim=True)
            b = torch.sum(b, dim=3, keepdim=True)
            return b

        def outlier_ratio(gtflow, predflow, mask, threshold=3.0, relative=0.05):
            diff = euclidean(gtflow - predflow) * mask
            threshold = torch.tensor(threshold).type_as(gtflow)
            if relative is not None:
                threshold_map = torch.max(threshold, euclidean(gt_flow) * relative)
                # outliers = tf.cast(tf.greater_equal(diff, threshold), tf.float32)
                outliers = diff > threshold_map
            else:
                # outliers = tf.cast(tf.greater_equal(diff, threshold), tf.float32)
                outliers = diff > threshold
            mask_s = batch_sum(mask)
            outliers_s = batch_sum(outliers)
            # print('outliers_s', outliers_s, 'mask_s', mask_s)
            ratio = outliers_s / (mask_s + 1e-6)
            return torch.mean(ratio.float())

        if mask is None:
            mask = torch.ones_like(predflow.narrow(1, 0, 1))
        if len(mask.shape) == 3:  # n,h,w
            mask = torch.unsqueeze(mask, 1)

        frac = outlier_ratio(gt_flow, predflow, mask, threshold, relative) * 100
        return frac

    @classmethod
    def demo(cls):
        from utils.tools import DIS_test
        model = DIS_test()
        bench = FlowEvalBench(eval_batch_size=1, eval_name='KITTI_2012', eval_datatype='nori', if_print_process=False)
        bench2 = FlowEvalBench(eval_batch_size=1, eval_name='KITTI_2015', eval_datatype='nori', if_print_process=False)
        bench(model)
        bench2(model)
        '''
        === eval time 63.2 s === eval_name='sintel_clean'
        sintel_clean, EPE = 4.77, NOC = 2.94, OCC = 15.01, F1 = 12.07
        === eval time 62.8 s === eval_name='sintel_final'
        sintel_final, EPE = 5.87, NOC = 4.05, OCC = 16.61, F1 = 16.89
        
        '''


class Training_eval_manager(tools.abstract_config):
    def __init__(self, **kwargs):
        self.validation_sets = ['flyingchairs', ]
        self.eval_batch_size = 8
        self.train_dir = '/data/Optical_Flow_all/train_RAFT'
        self.eval_datatype = 'base'  # base, nori
        self.eval_save_results = False
        self.if_print_process = False

        self.update(kwargs)
        self.bench_ls = []
        for i in self.validation_sets:
            print('=' * 3 + ' build evaluation bench: %s ' % i + '=' * 3)
            self.bench_ls.append(FlowEvalBench(eval_name=i, eval_batch_size=self.eval_batch_size, eval_datatype=self.eval_datatype, if_shuffle=False,
                                               eval_save_results=self.eval_save_results, if_print_process=self.if_print_process))
        self.best_score = 1e7
        self.best_results = None
        self.current_model_path = ''
        self.current_info_path = ''
        self.info_ls = {'param': kwargs, 'info_ls': []}

    def __call__(self, test_model: tools.abs_test_model, save_info: dict, save_name='raft'):
        results = []
        torch.set_grad_enabled(False)  # save the GPU memory
        for val_bench in self.bench_ls:
            results.append(val_bench(test_model))
        eval_score = sum([i[0] for i in results])
        save_info['results'] = results
        self.info_ls['info_ls'].append(save_info)
        if eval_score <= self.best_score:
            self.best_results = results
            self.best_score = eval_score
            if os.path.isfile(self.current_model_path):
                os.remove(self.current_model_path)

            if os.path.isfile(self.current_info_path):
                os.remove(self.current_info_path)

            save_path = os.path.join(self.train_dir, '%s.pth' % save_name)
            test_model.save_model(save_path)
            self.current_model_path = save_path

            info_save_path = os.path.join(self.train_dir, 'training_info.pkl')
            file_tools.pickle_saver.save_pickle(self.info_ls, info_save_path)
            self.current_info_path = info_save_path
        # save latest model
        latest_model_path = os.path.join(self.train_dir, '%s_latest.pth' % save_name)
        if os.path.isfile(latest_model_path):
            os.remove(latest_model_path)
        test_model.save_model(latest_model_path)
        torch.set_grad_enabled(True)  # save the GPU memory
        return eval_score, self.best_score

    def eval(self, test_model: tools.abs_test_model):
        results = []
        for val_bench in self.bench_ls:
            results.append(val_bench(test_model))
        eval_score = sum([i[0] for i in results])
        return eval_score, results

    def prin_best_results(self):
        for evscore, res_dict in self.best_results:
            name = res_dict['name']  # {'name':sintel,'EPE'0,'NOC':0,'OCC':0,'F1':0}
            evinfo = '%s(' % name
            for mk in sorted(list(res_dict.keys())):
                if mk != 'name':
                    evvalue = res_dict[mk]
                    evinfo += '%s=%.2f ' % (mk, evvalue)
            print(evinfo + ')')
