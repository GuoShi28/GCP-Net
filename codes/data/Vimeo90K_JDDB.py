'''
Vimeo90K dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
from utils import unprocess
import pdb

try:
    import mc  # import memcached
except ImportError:
    pass
logger = logging.getLogger('base')


class Vimeo90KDataset_RGGB(data.Dataset):
    '''
    Reading the training Vimeo90K dataset
    key example: 00001_0001 (_1, ..., _7)
    GT (Ground-Truth): 4th frame;
    LQ (Low-Quality): support reading N LQ frames, N = 1, 3, 5, 7 centered with 4th frame
    '''

    def __init__(self, opt):
        super(Vimeo90KDataset_RGGB, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.GT_root = opt['dataroot_GT']
        self.data_type = self.opt['data_type']
        # self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs

        #### determine the LQ frame list
        '''
        N | frames
        1 | 4
        3 | 3,4,5
        5 | 2,3,4,5,6
        7 | 1,2,3,4,5,6,7
        '''
        self.LQ_frames_list = []
        for i in range(opt['N_frames']):
            self.LQ_frames_list.append(i + (9 - opt['N_frames']) // 2)
        # pdb.set_trace()
        #### directly load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
            logger.info('Using lmdb meta info for cache keys.')
        elif opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
        else:
            raise ValueError(
                'Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def __getitem__(self, index):
        if self.data_type == 'mc':
            self._ensure_memcached()
        elif self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()

        # pdb.set_trace()
        # scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')
        #### get the GT image (as the center frame)
        img_GT = util.read_img(self.GT_env, key + '_4', (3, 256, 448))
        img_GT = util.BGR2RGB(img_GT) # RGB
        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT))
        img_GT = img_GT.permute(2, 0, 1) # Re-Permute the tensor back CxHxW format
        img_GT, metadata = unprocess.unprocess_gt(img_GT)
        
        ## Random noise level
        shot_noise, read_noise = unprocess.random_noise_levels_kpn()
        # pdb.set_trace()
        #### get LQ images
        # LQ_size_tuple = (3, 64, 112) if self.LR_input else (3, 256, 448)
        LQ_size_tuple = (3, 256, 448)
        img_LQ_l = []
        for v in self.LQ_frames_list:
            # print(key+'_'+str(v))
            #img_LQ = util.read_img(self.LQ_env, key + '_{}'.format(v), LQ_size_tuple)
            img_LQ = util.read_img(self.GT_env, key + '_{}'.format(v), LQ_size_tuple)
            img_LQ = util.BGR2RGB(img_LQ) # RGB
            img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ))
            img_LQ = img_LQ.permute(2, 0, 1) # Re-Permute the tensor back CxHxW format
            img_LQ, _ = unprocess.unprocess_meta_gt(img_LQ, metadata['rgb_gain'], \
                metadata['red_gain'], \
                metadata['blue_gain'], metadata['rgb2cam'], metadata['cam2rgb'])

            img_LQ_l.append(img_LQ)

        img_noise_map = []
        if self.opt['phase'] == 'train':
            # LQ_size_tuple_new = (4, 128, 224)
            LQ_size_tuple = (3, 256, 448)
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_LQ_l = [v[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size] for v in img_LQ_l]
            img_GT = img_GT[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]

            img_LQ_l = [unprocess.mosaic(v) for v in img_LQ_l]
            img_LQ_l = [unprocess.add_noise(v, shot_noise, read_noise) for v in img_LQ_l]
            img_LQ_l = [v.clamp(0.0, 1.0) for v in img_LQ_l]
            # img_noise_map = [torch.sqrt(shot_noise * v + read_noise**2) for v in img_LQ_l]
            img_noise_map = [shot_noise * v + read_noise for v in img_LQ_l]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = torch.stack(img_LQ_l, axis=0)
        img_NMaps = torch.stack(img_noise_map, axis=0)

        inputs = {'LQs': img_LQs, 'NMaps': img_NMaps, 'GT': img_GT, 'key': key}
        inputs.update(metadata)

        return inputs

    def __len__(self):
        return len(self.paths_GT)

if __name__ == '__main__':

   pass
