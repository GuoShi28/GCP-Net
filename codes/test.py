'''
Test Vid4 and REDS4 datasets
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import pdb
import utils.util as util
import data.util as data_util
import utils.process as process
import utils.unprocess as unprocess
import torch.nn.functional as F

def read_rggb_img_seq_opts_joint(path, metadata):
    """Read a sequence of images from a given folder path
    Args:
        path (list/str): list of image paths/image folder path

    Returns:
        imgs (Tensor): size (T, C, H, W), RGGB, [0, 1]
    """
    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))
    img_l = []
    img_noise = []
    img_noise_map = []
    count = 1
    for v in img_path_l:
        print(v)
        temp = data_util.read_img(None, v)
        temp = data_util.BGR2RGB(temp)
        # pdb.set_trace()
        temp = torch.from_numpy(np.ascontiguousarray(temp))
        temp = temp.permute(2, 0, 1) # Re-Permute the tensor back CxHxW format
        
        temp, _ = unprocess.unprocess_meta_gt(temp, metadata['rgb_gain'], \
            metadata['red_gain'], \
            metadata['blue_gain'], metadata['rgb2cam'], \
            metadata['cam2rgb'])

        img_l.append(temp)
        shot_noise, read_noise = 6.4e-3, 2e-2
        # shot_noise, read_noise = 2.5e-3, 1e-2

        temp = unprocess.mosaic(temp)
        temp = unprocess.add_noise_test(temp, shot_noise, read_noise, count)
        count = count + 1
        temp = temp.clamp(0.0, 1.0)
        temp_np = shot_noise * temp + read_noise

        img_noise.append(temp)
        img_noise_map.append(temp_np)

    # stack to Torch tensor
    imgs = torch.stack(img_l, axis=0)
    imgs_n = torch.stack(img_noise, axis=0)
    img_np = torch.stack(img_noise_map, axis=0)
    return imgs, imgs_n, img_np

def main():
    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    data_mode = 'REDS4'  # Vid4
    # data_mode = 'Vid4'
    # Vid4: SR
    flip_test = False
    ############################################################################
    ############################################################################
    #### model
    import models.archs.arch_gcpnet as arch_gcpnet
    model = arch_gcpnet.GCPNet(nf=64,\
                groups=8, in_channel=1, output_channel=3)
    model_path = '../experiments/gcpnet_model/600000_G.pth'
    
    # -----------------Test noise level setting ----------------------------------------------
    ## Metadata
    xyz2cam = torch.FloatTensor([[1.0234, -0.2969, -0.2266],
                                [-0.5625, 1.6328, -0.0469],
                                [-0.0703, 0.2188, 0.6406]])
    rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)
    # Normalizes each row.
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
    cam2rgb = torch.inverse(rgb2cam)
    rgb_gains = 1.0 

    # Red and blue gains represent white balance.
    red_gains  =  torch.FloatTensor([2.0])
    blue_gains =  torch.FloatTensor([1.7])

    metadata = {
        'cam2rgb': cam2rgb,
        'rgb2cam': rgb2cam,
        'rgb_gain': rgb_gains,
        'red_gain': red_gains,
        'blue_gain': blue_gains,
    }

    N_in = 5
    #### dataset
    if data_mode == 'Vid4':
        GT_dataset_folder = '../datasets/Vid4/GT'
    else:
        GT_dataset_folder = '../datasets/REDS4/GT'
    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    if data_mode == 'Vid4' or data_mode == 'sharp_bicubic':
        padding = 'new_info'
    else:
        padding = 'replicate'
    save_imgs = True

    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {}'.format(data_mode))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)

    ### 
    params = sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])
    print('Parameter count:{}'.format(params))
    #### Merge end #####
    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    avg_ssim_center_l = []
    subfolder_name_l = []

    subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
    # for each subfolder
    for subfolder_GT in subfolder_GT_l:
        subfolder_name = osp.basename(subfolder_GT)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder_GT, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)
        #### read LQ and GT images
        img_GT_l, imgs_LQ, imgs_NMap = \
            read_rggb_img_seq_opts_joint(img_path_l, metadata)

        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0
        avg_ssim_center = 0
        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
            imgs_nmap_in = imgs_NMap.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
            
            # -------> if the CUDA is not out of memory
            output = util.single_forward(model, imgs_in, imgs_nmap_in)
            
            output, _ = process.process_test(output, metadata['red_gain'].unsqueeze(0), \
                metadata['blue_gain'].unsqueeze(0), metadata['cam2rgb'].unsqueeze(0))
            output = output.squeeze(0)
            output = util.tensor2img(output)
            output = np.clip(output, 0, 255)
           
            # calculate PSNR
            output = output / 255.
            # GT = np.copy(img_GT_l[img_idx])
            GT = img_GT_l[img_idx:img_idx+1, :, :, :]
            GT, GT_show = process.process_test(GT, metadata['red_gain'].unsqueeze(0), \
                metadata['blue_gain'].unsqueeze(0), metadata['cam2rgb'].unsqueeze(0))
            GT = GT.squeeze(0)
            GT = np.copy(GT)
            GT = np.transpose(GT, (1, 2, 0))
            GT = GT[..., ::-1]
            
            if save_imgs:
                output_results = output
                output_results = (output_results * 255.0).round()
                output_results = output_results.astype(np.uint8)
                cv2.imwrite(osp.join(save_subfolder, '{}_gcpnet.png'.format(img_name)), output_results)
        
            output, GT = util.crop_border([output, GT], crop_border)
            crt_psnr = util.calculate_psnr(output * 255, GT * 255)
            crt_ssim = util.calculate_ssim(output * 255, GT * 255)
            logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB SSIM: {:.6f}'.format(img_idx + 1, img_name, crt_psnr, crt_ssim))

            if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
                avg_psnr_center += crt_psnr
                avg_ssim_center += crt_ssim
                N_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                N_border += 1

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_ssim_center = avg_ssim_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_ssim_center_l.append(avg_ssim_center)
        avg_psnr_border_l.append(avg_psnr_border)

        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                    'Center PSNR: {:.6f} dB for {} frames; '
                    'Center SSIM: {:.6f} for {} frames; '
                    'Border PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_psnr,
                                                                   (N_center + N_border),
                                                                   avg_psnr_center, N_center,
                                                                   avg_ssim_center, N_center,
                                                                   avg_psnr_border, N_border))

    logger.info('################ Tidy Outputs ################')
    for subfolder_name, psnr, psnr_center, psnr_border, ssim_center in zip(subfolder_name_l, avg_psnr_l,
                                                              avg_psnr_center_l, avg_psnr_border_l, avg_ssim_center_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                    'Center PSNR: {:.6f} dB. '
                    'Center SSIM: {:.6f}. '
                    'Border PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center, ssim_center,
                                                     psnr_border))
    logger.info('################ Final Results ################')
    logger.info('Data: {}'.format(data_mode))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB, center SSIM: {:.6f}. Border PSNR: {:.6f} dB.'.format(
                    sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_GT_l),
                    sum(avg_psnr_center_l) / len(avg_psnr_center_l),
                    sum(avg_ssim_center_l) / len(avg_ssim_center_l),
                    sum(avg_psnr_border_l) / len(avg_psnr_border_l)))


if __name__ == '__main__':
    main()
