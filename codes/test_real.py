'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import math
import h5py
import scipy.io as sio

import utils.util as util
import data.util as data_util
import utils.process as process
import utils.unprocess as unprocess
import torch.nn.functional as F
import pdb

def read_raw_img_seq(path):
    img_path_l = path
    img_l = []
    # pdb.set_trace()
    for idx, v in enumerate(img_path_l):
        print(v)
        img_denoise = h5py.File(v, 'r')
        buffer_np = np.float32(np.array(img_denoise['im']).T)
        buffer_np_4 = np.zeros((int(buffer_np.shape[0]/2), \
            int(buffer_np.shape[1]/2), 4), dtype=buffer_np.dtype)
        buffer_np_4[:,:,0] = buffer_np[0::2, 0::2]
        buffer_np_4[:,:,1] = buffer_np[0::2, 1::2]
        buffer_np_4[:,:,2] = buffer_np[1::2, 0::2]
        buffer_np_4[:,:,3] = buffer_np[1::2, 1::2]
        img_l.append(buffer_np_4)
        if idx == 2:
            meta_data = img_denoise['meta']
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float() # N * 4 * H * W

    return imgs, meta_data


def main():
    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    data_mode = 'real_data'  
    flip_test = False
    ############################################################################
    #### model
    ## Paper model
    import models.archs.arch_gcpnet as arch_gcpnet
    model = arch_gcpnet.GCPNet(nf=64,\
                groups=8, in_channel=1, output_channel=3)
    model_path = '../experiments/gcpnet_model/600000_G.pth'
    
    N_in = 5
    #### dataset
    test_dataset_folder = '../datasets/SC_burst/'

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    padding = 'new_info'

    save_imgs = True

    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    print(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    for fol_in, subfolder_L in enumerate(subfolder_l):
        subfolder_name = osp.basename(subfolder_L)

        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder_L, '*.mat')))
        img_path_l = img_path_l[0:5]
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ, metadata = read_raw_img_seq(img_path_l)

        ### read meta data
        ColorMatrix = np.array(metadata['ColorMatrix2'])
        WhiteParam = np.array(metadata['AsShotNeutral'])
        UnknoTag = metadata['UnknownTags']
        TagValue = UnknoTag['Value']
        TagNoise = TagValue[0,7]
        NoisePara = np.array(UnknoTag[TagNoise])
        shot_noise = torch.from_numpy(NoisePara[0]).float()
        read_noise = torch.from_numpy(np.sqrt(NoisePara[1])).float()
        ## Metadata
        xyz2cam = torch.FloatTensor(ColorMatrix)
        xyz2cam = torch.reshape(xyz2cam, (3, 3))
        rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
        
        rgb2cam = torch.mm(xyz2cam, rgb2xyz)
        # Normalizes each row.
        rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
        cam2rgb = torch.inverse(rgb2cam)
        rgb_gains = 1.0 
        # Red and blue gains represent white balance.
        red_gains  =  torch.FloatTensor(1.0/WhiteParam[0])
        blue_gains =  torch.FloatTensor(1.0/WhiteParam[2])
        metadata = {
            'cam2rgb': cam2rgb,
            'rgb_gain': rgb_gains,
            'red_gain': red_gains,
            'blue_gain': blue_gains,
        }

        W = imgs_LQ.shape[2]
        H = imgs_LQ.shape[3]
        
        imgs_in = imgs_LQ # T, C(4), H, W
        imgs_in = imgs_in.view(1, imgs_in.shape[0], imgs_in.shape[1], imgs_in.shape[2], imgs_in.shape[3])
        img_in_nmap = shot_noise * imgs_in + read_noise
        patch_size = 256
        patch_extend = 16
        extend_size = patch_size + 2*patch_extend

        origin_w = imgs_in.shape[3]
        origin_h = imgs_in.shape[4]
        w_res = origin_w % patch_size
        h_res = origin_h % patch_size

        imgs_in = imgs_in.numpy()
        imgs_in_pad = np.pad(imgs_in, ((0, 0), (0, 0), (0, 0), (0, patch_size - w_res), (0, patch_size - h_res)), 'constant')
        imgs_in_pad_pad = np.pad(imgs_in_pad, ((0, 0), (0, 0), (0, 0), (patch_extend, patch_extend), (patch_extend, patch_extend)), 'constant')

        img_in_nmap = img_in_nmap.numpy()
        img_in_nmap_pad = np.pad(img_in_nmap, ((0, 0), (0, 0), (0, 0), (0, patch_size - w_res), (0, patch_size - h_res)), 'constant')
        img_in_nmap_pad_pad = np.pad(img_in_nmap_pad, ((0, 0), (0, 0), (0, 0), (patch_extend, patch_extend), (patch_extend, patch_extend)), 'constant')
        
        imgs_in = torch.from_numpy(imgs_in)
        imgs_in_pad = torch.from_numpy(imgs_in_pad)
        imgs_in_pad_pad = torch.from_numpy(imgs_in_pad_pad)

        img_in_nmap = torch.from_numpy(img_in_nmap)
        img_in_nmap_pad = torch.from_numpy(img_in_nmap_pad)
        img_in_nmap_pad_pad = torch.from_numpy(img_in_nmap_pad_pad)

        new_w = imgs_in_pad.shape[3]
        new_h = imgs_in_pad.shape[4]
        w_num = new_w / patch_size
        h_num = new_h / patch_size

        denoised_img = np.zeros((3, int(new_w*2), int(new_h*2)), dtype=np.float)
        for w_index in range(math.floor(w_num)):
            for h_index in range(math.floor(h_num)):
                print('w = '+str(w_index))
                print('h = '+str(h_index))
                start_x = w_index * patch_size
                end_x = start_x + patch_size - 1
                if end_x > new_w-1:
                    end_x = new_w-1
                start_y = h_index * patch_size
                end_y = start_y + patch_size - 1
                if end_y > new_h-1:
                    end_y = new_h-1
                image_patch = imgs_in_pad[:,:,:,start_x:end_x+1, start_y:end_y+1]
                image_patch_pad = imgs_in_pad_pad[:,:,:,start_x:end_x+2*patch_extend+1,start_y:end_y+2*patch_extend+1]

                nmpa_patch = img_in_nmap_pad[:,:,:,start_x:end_x+1, start_y:end_y+1]
                nmap_patch_pad = img_in_nmap_pad_pad[:,:,:,start_x:end_x+2*patch_extend+1,start_y:end_y+2*patch_extend+1]

                image_patch = image_patch.cuda()
                image_patch_pad = image_patch_pad.cuda()
                nmpa_patch = nmpa_patch.cuda()
                nmap_patch_pad = nmap_patch_pad.cuda()
                # --- Paper our model ---
                output = util.single_forward(model, image_patch_pad, nmap_patch_pad)
                # convert linear RGB to sRGB
                output, _ = process.process_test(output, metadata['red_gain'].unsqueeze(0), \
                    metadata['blue_gain'].unsqueeze(0), metadata['cam2rgb'].unsqueeze(0))
                
                start_x2 = w_index * 2 * patch_size
                end_x2 = start_x2 + 2 * patch_size - 1
                start_y2 = h_index * 2 * patch_size
                end_y2 = start_y2 + 2 * patch_size - 1
                output = output.squeeze().float().cpu().numpy()
                output = output[:, 2*patch_extend:-2*patch_extend, 2*patch_extend:-2*patch_extend]
                denoised_img[:, start_x2:end_x2+1, start_y2:end_y2+1] = output
                      
        denoised_img = denoised_img[:, 0:int(2*origin_w), 0:int(2*origin_h)]

        output_results = denoised_img
        output_results = np.clip(output_results, 0.0, 1.0)
        output_results = np.transpose(output_results, (1, 2, 0))
        output_results_save = output_results
        
        if save_imgs:
            eps = 1e-8
            output_results = np.clip(output_results, 0.0, 1.0)
            output_results_temp = output_results
            output_results = output_results[..., ::-1]
            output_results = (output_results * 255.0).round()
            output_results = output_results.astype(np.uint8)
            cv2.imwrite(osp.join(save_subfolder, 'gcpnet.png'), output_results)

            temp = imgs_in[0, 2, :, :, :]
            temp = temp.squeeze().float().cpu().numpy()
            noise_input_full = np.zeros((int(temp.shape[1]*2), int(temp.shape[2]*2)), np.float)
            noise_input_full[0::2, 0::2] = temp[0, :, :]
            noise_input_full[0::2, 1::2] = temp[1, :, :]
            noise_input_full[1::2, 0::2] = temp[2, :, :]
            noise_input_full[1::2, 1::2] = temp[3, :, :]
            noise_input_full = util.bayer2bgr(noise_input_full) # demosaicked by using traditional method
            noise_input_full = noise_input_full[..., [2, 1, 0]]
            noise_input_full = noise_input_full.transpose(2, 0, 1)
            noise_input_full = torch.from_numpy(noise_input_full).float()
            noise_input_full = torch.unsqueeze(noise_input_full, 0)
            noise_input_full, _ = process.process_test(noise_input_full, metadata['red_gain'].unsqueeze(0), \
                metadata['blue_gain'].unsqueeze(0), metadata['cam2rgb'].unsqueeze(0))
            noise_input_full = noise_input_full.squeeze(0)

            noise_input_full = np.copy(noise_input_full)
            noise_input_full = np.transpose(noise_input_full, (1, 2, 0))
            noise_input_full = noise_input_full[..., ::-1]

            noise_input_full = (noise_input_full * 255.0).round()
            noise_input_full = noise_input_full.astype(np.uint8)
            cv2.imwrite(osp.join(save_subfolder, 'noise_rgb.png'), noise_input_full)
            

if __name__ == '__main__':
    main()
