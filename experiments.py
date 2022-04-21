from train import train_irrpwc
import os


# When there are many experimental models, putting the model in the GitHub folder will take up a lot of storage space.
# I will update the models in my experiments later.
class pretrain_models():
    class abs_mp():
        def __init__(self, path='', info='', note=''):
            self.path = path
            self.info = info
            self.note = note

    class chairs_model():
        def __init__(self):
            self.ph_sm_bdw = pretrain_models.abs_mp(path='./pretrain_models/chairs/irrpwc_ph_sm_bdw_EPE=3.43.pth',
                                                    info='flyingchairs(EPE=3.43 )',
                                                    note='single forward flow, only photo loss and smooth loss, augmentation is not used')
            self.ph_sm_bdw_1 = pretrain_models.abs_mp(path='./pretrain_models/chairs/irrpwc_ph_sm_bdw_EPE=3.23.pth',
                                                      info='flyingchairs(EPE=3.23 )',
                                                      note='single forward flow, only photo loss and smooth loss, augmentation is used, lr=1e-5')


# /usr/bin/python3 experiments.py
def flying_chairs_experiment():
    stage = 1
    # the first stage: training using only photo loss and smooth loss, training from scratch, flyingchairs(EPE=3.43 )
    if stage == 1:
        param = {
            'if_show_training': False,
            'train_dir': '/data/Optical_Flow_all/train_OIFlow_irrpwc_chairs_1',
            # ==== dataset
            'train_set': 'flyingchairs',
            'validation_sets': ('flyingchairs',),
            'eval_batch_size': 8,
            'image_size': (384, 512),

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
            'photo_loss_type': 'abs_robust',  # abs_robust, charbonnier,L1,SSIM(can not use, only for occlusion checking is enabled)
            'smooth_loss_type': 'delta',  # 'edge' or 'delta'
            'smooth_loss_weight_1': 0.1,
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
            'equivariant_rotate': (-0.2, 0.2, -0.015, 0.015),  # (-0.01, 0.01, -0.01, 0.01)
            'equivariant_squeeze': (0.86, 1.16, 1.0, 1.0),  # (1.0, 1.0, 1.0, 1.0)
            'equivariant_trans': (0.2, 0.015),  # (0.04, 0.005)
            'equivariant_vflip': False,
            'equivariant_zoom': (1.0, 1.5, 0.985, 1.015),  # (1.0, 1.4, 0.99, 1.01)
        }
    # the first stage: decrease the learning rate and finetune, only photo loss and smooth loss, augmentation is used, EPE=3.23
    elif stage == 1.1:
        param = {
            'if_show_training': False,
            'train_dir': '/data/Optical_Flow_all/train_OIFlow_irrpwc_chairs_1-1',
            # ==== dataset
            'train_set': 'flyingchairs',
            'validation_sets': ('flyingchairs',),
            'eval_batch_size': 8,
            'image_size': (384, 512),

            # ==== model
            'pretrain_path': pretrain_models.chairs_model().ph_sm_bdw.path,  # using model from stage 1
            'if_froze_pwc': False,
            'if_use_correlation_pytorch': False,  # just for debug

            # ==== training
            'lr': 1e-5,  # change
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
            'photo_loss_type': 'abs_robust',  # abs_robust, charbonnier,L1,SSIM(can not use, only for occlusion checking is enabled)
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
            'aug_vertical_prob': 0.1,
            # photo metric and occlusion aug
            'aug_color_prob': 0.5,
            'aug_color_asymmetric_prob': 0.3,
            'aug_eraser_prob': 0.2,

            # ==== test
            'eval_save_results': False,
            'eval_some_save_results': False,
            # ==== eq loss parameters
            'equivariant_loss_mask_norm': False,
            'eq_loss_weight': 0.1,
            'equivariant_add_noise': False,
            'equivariant_hflip': True,
            'equivariant_rotate': (-0.2, 0.2, -0.015, 0.015),  # (-0.01, 0.01, -0.01, 0.01)
            'equivariant_squeeze': (0.86, 1.16, 1.0, 1.0),  # (1.0, 1.0, 1.0, 1.0)
            'equivariant_trans': (0.2, 0.015),  # (0.04, 0.005)
            'equivariant_vflip': False,
            'equivariant_zoom': (1.0, 1.5, 0.985, 1.015),  # (1.0, 1.4, 0.99, 1.01)
        }
    # the second stage: bidirection flow and add occlusion checking(forward-backward check, range map check is not implemented) todo now training
    elif stage == 2:
        param = {
            'if_show_training': False,
            'train_dir': '/data/Optical_Flow_all/train_OIFlow_irrpwc_chairs_2',
            # ==== dataset
            'train_set': 'flyingchairs',
            'validation_sets': ('flyingchairs',),
            'eval_batch_size': 8,
            'image_size': (384, 512),

            # ==== model
            'pretrain_path': pretrain_models.chairs_model().ph_sm_bdw_1.path,  # using model from stage 1
            'if_froze_pwc': False,
            'if_use_correlation_pytorch': False,  # just for debug

            # ==== training
            'lr': 1e-4,  # change
            'wdecay': 1e-5,
            'num_epochs': 2000,
            'batch_size': 6,
            'optim_gamma': 1,
            'print_every': 50,
            'eval_every': 500,

            # ==== loss
            'train_func': 'train_bi_direction_photo_occ_smooth',
            'photo_loss_weight': 1,
            'norm_photo_loss_scale': 0.5,  # the original image scale is 0~255
            'photo_loss_type': 'abs_robust',  # abs_robust, charbonnier,L1,SSIM(can not use, only for occlusion checking is enabled)
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
            'aug_vertical_prob': 0.1,
            # photo metric and occlusion aug
            'aug_color_prob': 0.5,
            'aug_color_asymmetric_prob': 0.3,
            'aug_eraser_prob': 0.2,

            # ==== test
            'eval_save_results': False,
            'eval_some_save_results': False,
            # ==== eq loss parameters
            'equivariant_loss_mask_norm': False,
            'eq_loss_weight': 0.1,
            'equivariant_add_noise': False,
            'equivariant_hflip': True,
            'equivariant_rotate': (-0.2, 0.2, -0.015, 0.015),  # (-0.01, 0.01, -0.01, 0.01)
            'equivariant_squeeze': (0.86, 1.16, 1.0, 1.0),  # (1.0, 1.0, 1.0, 1.0)
            'equivariant_trans': (0.2, 0.015),  # (0.04, 0.005)
            'equivariant_vflip': False,
            'equivariant_zoom': (1.0, 1.5, 0.985, 1.015),  # (1.0, 1.4, 0.99, 1.01)
        }
    # the third stage: add self-supervision(I call Equivariant loss or eq loss here) todo
    elif stage == 3:
        raise ValueError('not implemented')
    # the forth stage: add occlusion inpaining todo not implemented here, please wait for some time.
    elif stage == 4:
        raise ValueError('not implemented')
    else:
        raise ValueError('not implemented')
    trainer = train_irrpwc(**param)
    trainer()


def do_eval():
    model_path = ''
    param = {
        'eval_save_results': False,
        'eval_some_save_results': False,

        'pretrain_path': model_path,
        'if_use_correlation_pytorch': False,  # just for debug
        'train_dir': os.path.split(model_path)[0],  # also used as result save dir

        # ====== dataset conf
        'validation_sets': ('flyingchairs',),  # flyingchairs,sintel_clean,sintel_final,KITTI_2012,KITTI_2015
        'eval_batch_size': 1,  # note that: on KITTI eval batch size should be 1 for the image size may change
    }
    trainer = train_irrpwc(**param)
    trainer.eval()


if __name__ == '__main__':
    flying_chairs_experiment()
