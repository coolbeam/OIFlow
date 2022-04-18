from train import train_irrpwc


# /usr/bin/python3 experiments.py
def flying_chairs_experiment():
    stage = 1
    # the first stage: training using only photo loss and smooth loss
    if stage == 1:
        param = {
            'if_show_training': False,
            'train_dir': '/data/Optical_Flow_all/train_OIFlow_irrpwc',
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
    # the second stage: add occlusion checking todo
    elif stage == 2:
        raise ValueError('not implemented')
    # the third stage: add self-supervision(I call eq loss here) todo
    elif stage == 3:
        raise ValueError('not implemented')
    # the forth stage: add occlusion inpaining todo not implemented here, please wait for some time.
    elif stage == 4:
        raise ValueError('not implemented')
    else:
        raise ValueError('not implemented')
    trainer = train_irrpwc(**param)
    trainer()


if __name__ == '__main__':
    flying_chairs_experiment()
