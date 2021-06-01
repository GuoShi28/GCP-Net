import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import pdb
from utils import unprocess, process
import glob
import numpy as np


class VideoTestDataset_RGGB(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(VideoTestDataset_RGGB, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        # GS: No need LR
        #self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.GT_root = opt['dataroot_GT']
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        
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
        rgb_gains = 1.0 / 0.5

        # Red and blue gains represent white balance.
        red_gains  =  torch.FloatTensor([2.0])
        blue_gains =  torch.FloatTensor([1.7])

        self.metadata = {
            'cam2rgb': cam2rgb,
            'rgb2cam': rgb2cam,
            'rgb_gain': rgb_gains,
            'red_gain': red_gains,
            'blue_gain': blue_gains,
        }

        #### Generate data info and cache data
        self.imgs_LQ, self.imgs_GT, self.imgs_LQNM = {}, {}, {}
        if opt['name'].lower() in ['vid4', 'reds4']:
            # subfolders_LQ = util.glob_file_list(self.LQ_root)
            subfolders_GT = util.glob_file_list(self.GT_root)
            # for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            for subfolder_GT in subfolders_GT:
                subfolder_name = osp.basename(subfolder_GT)
                # img_paths_LQ = util.glob_file_list(subfolder_LQ)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                max_idx = len(img_paths_GT)
                assert max_idx == len(
                    img_paths_GT), 'Different number of images in LQ and GT folders'
                # self.data_info['path_LQ'].extend(img_paths_LQ)
                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                # pdb.set_trace()
                if self.cache_data:
                    # self.imgs_LQ[subfolder_name] = util.read_img_seq(img_paths_LQ)
                    self.imgs_GT[subfolder_name], self.imgs_LQ[subfolder_name], self.imgs_LQNM[subfolder_name] = \
                        read_rggb_img_seq_opts_joint(img_paths_GT, self.metadata)

        elif opt['name'].lower() in ['vimeo90k-test']:
            pass  # TODO
        else:
            raise ValueError(
                'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

    def __getitem__(self, index):
        # path_LQ = self.data_info['path_LQ'][index]
        # path_GT = self.data_info['path_GT'][index]
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        if self.cache_data:
            select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],
                                               padding=self.opt['padding'])
            imgs_LQ = self.imgs_LQ[folder].index_select(0, torch.LongTensor(select_idx))
            imgs_LQNM = self.imgs_LQNM[folder].index_select(0, torch.LongTensor(select_idx))
            img_GT = self.imgs_GT[folder][idx]
        else:
            pass  # TODO

        inputs = {
            'LQs': imgs_LQ,
            'GT': img_GT,
            'NMaps': imgs_LQNM,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border
        }
        inputs.update(self.metadata)

        return inputs

    def __len__(self):
        return len(self.data_info['path_GT'])


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
    # img_l = [read_img(None, v) for v in img_path_l]
    img_l = []
    img_noise = []
    img_noise_map = []
    # pdb.set_trace()
    for v in img_path_l:
        print(v)
        temp = util.read_img(None, v)
        temp = util.BGR2RGB(temp)
        # pdb.set_trace()
        temp = torch.from_numpy(np.ascontiguousarray(temp))
        temp = temp.permute(2, 0, 1) # Re-Permute the tensor back CxHxW format
        
        temp, _ = unprocess.unprocess_meta_gt(temp, metadata['rgb_gain'], \
            metadata['red_gain'], \
            metadata['blue_gain'], metadata['rgb2cam'], \
            metadata['cam2rgb'])

        img_l.append(temp)
        shot_noise, read_noise = 6.4e-3, 2e-2 

        temp = unprocess.mosaic(temp)
        temp = unprocess.add_noise(temp, shot_noise, read_noise)
        temp = temp.clamp(0.0, 1.0)
        # temp_np = torch.sqrt(shot_noise * temp + read_noise**2)
        temp_np = shot_noise * temp + read_noise

        img_noise.append(temp)
        img_noise_map.append(temp_np)

        '''
        cv2.imshow('img', temp)
        cv2.waitKey(0)
        '''
    # stack to Torch tensor
    imgs = torch.stack(img_l, axis=0)
    imgs_n = torch.stack(img_noise, axis=0)
    img_np = torch.stack(img_noise_map, axis=0)
    # pdb.set_trace()
    return imgs, imgs_n, img_np
