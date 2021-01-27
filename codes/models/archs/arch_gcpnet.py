''' network architecture for backbone '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import models.archs.arch_util as arch_util
import numpy as np
import math
import pdb
from torch.nn.modules.utils import _pair
from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN

class SimpleBlock(nn.Module):
    def __init__(self, depth=3, n_channels=64, input_channels=3, output_channel=64, kernel_size=3):
        super(SimpleBlock, self).__init__()
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=output_channel, kernel_size=kernel_size, padding=padding, bias=False))
        self.simple_block = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.simple_block(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class DualABlock(nn.Module):
    def __init__(self, res_num=5, n_channels=64, input_channels=3, output_channel=64, kernel_size=3):
        super(DualABlock, self).__init__()
        padding = 1
        self.res_num = res_num
        self.square_conv = nn.Conv2d(in_channels=input_channels, out_channels=n_channels, \
            kernel_size=(kernel_size, kernel_size), padding=(padding, padding), bias=False)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.extract_conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, \
            kernel_size=kernel_size, padding=padding, bias=True)
       
        self.res_block1 = SimpleBlock(depth=2, n_channels=n_channels, input_channels=n_channels, \
                output_channel=n_channels, kernel_size=3)  # 64, H, W
        self.res_block2 = SimpleBlock(depth=2, n_channels=n_channels, input_channels=n_channels, \
                output_channel=n_channels, kernel_size=3) # 64, H, W

        
        self.down = nn.Conv2d(in_channels=n_channels, out_channels=int(n_channels/2), kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=int(n_channels/2), out_channels=n_channels, kernel_size=1, stride=1, bias=True)
        self.spatial_att = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=7, stride=1, padding=3,bias=True)

        self._initialize_weights()

    def forward(self, x):
        x_temp = self.square_conv(x)
        x_temp = self.relu(x_temp)
        x_temp = self.extract_conv(x_temp)
        x_temp = x + x_temp

        x_temp2 = self.res_block1(x_temp)
        x_temp = x_temp + x_temp2
        x_temp2 = self.res_block2(x_temp)
        x_temp = x_temp + x_temp2

        # channel attention
        x_se = F.avg_pool2d(x_temp, kernel_size=(x_temp.size(2), x_temp.size(3)))
        x_se = self.down(x_se)
        x_se = self.relu(x_se)
        x_se = self.up(x_se)
        x_se = F.sigmoid(x_se)
        x_se = x_se.repeat(1, 1, x_temp.size(2), x_temp.size(3))
        # spatial attention
        x_sp = F.sigmoid(self.spatial_att(x_temp))
        
        x_temp = x_temp + x_temp * x_se + x_temp * x_sp 
        
        return x_temp

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class GCABlock(nn.Module):
    def __init__(self, res_num=5, n_channels=64, input_channels=3, output_channel=64, kernel_size=3):
        super(GCABlock, self).__init__()
        padding = 1
        self.res_num = res_num
        self.square_conv = nn.Conv2d(in_channels=input_channels, out_channels=n_channels, \
            kernel_size=kernel_size, padding=padding, bias=False)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.extract_conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, \
            kernel_size=kernel_size, padding=padding, bias=True)

        self.res_block1 = SimpleBlock(depth=2, n_channels=n_channels, input_channels=n_channels, \
                output_channel=n_channels, kernel_size=3)  # 64, H, W
        self.res_block2 = SimpleBlock(depth=2, n_channels=n_channels, input_channels=n_channels, \
                output_channel=n_channels, kernel_size=3) # 64, H, W

        
        self.down = nn.Conv2d(in_channels=n_channels, out_channels=int(n_channels/2), kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=int(n_channels/2), out_channels=n_channels, kernel_size=1, stride=1, bias=True)
        self.spatial_att = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=7, stride=1, padding=3,bias=True)

        self._initialize_weights()

    def forward(self, x, guided_lam, guided_beta):
        x_temp = self.square_conv(x)
        x_temp = x_temp.mul(guided_lam) + guided_beta
        
        x_temp = self.relu(x_temp)
        x_temp = self.extract_conv(x_temp)
        x_temp = x + x_temp
        
        x_temp2 = self.res_block1(x_temp)
        x_temp = x_temp + x_temp2
        x_temp2 = self.res_block2(x_temp)
        x_temp = x_temp + x_temp2

        # channel attention
        x_se = F.avg_pool2d(x_temp, kernel_size=(x_temp.size(2), x_temp.size(3)))
        x_se = self.down(x_se)
        x_se = self.relu(x_se)
        x_se = self.up(x_se)
        x_se = F.sigmoid(x_se)
        x_se = x_se.repeat(1, 1, x_temp.size(2), x_temp.size(3))
        # spatial attention
        x_sp = F.sigmoid(self.spatial_att(x_temp))
        
        x_temp = x_temp + x_temp * x_se + x_temp * x_sp 
        
        return x_temp

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleLSTM, self).__init__()
        self.nf = input_dim
        self.hf = hidden_dim
        self.conv = nn.Conv2d(self.nf+self.hf, 4*self.hf, 3, 1, 1, bias=True)

    def forward(self, input_tensor, h_cur, c_cur):
        if h_cur is None:
            tensor_size = (input_tensor.size(2),input_tensor.size(3))
            h_cur = self._init_hidden(batch_size=input_tensor.size(0),tensor_size=tensor_size)
            c_cur = self._init_hidden(batch_size=input_tensor.size(0),tensor_size=tensor_size)
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hf, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
    
    def _init_hidden(self, batch_size, tensor_size):
        height, width = tensor_size
        return torch.zeros(batch_size, self.hf, height, width).cuda()

class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''
    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_rnn = SimpleLSTM(nf, int(nf/2))
        self.L3_rnn_conv = nn.Conv2d(nf+int(nf/2), nf, 3, 1, 1, bias=True)

        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_rnn = SimpleLSTM(nf, int(nf/2))
        self.L2_rnn_conv = nn.Conv2d(nf+int(nf/2), nf, 3, 1, 1, bias=True)

        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_rnn = SimpleLSTM(nf, int(nf/2))
        self.L1_rnn_conv = nn.Conv2d(nf+int(nf/2), nf, 3, 1, 1, bias=True)

        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l, guided_nbr_fea_l, guided_ref_fea_l, h_cur, c_cur):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        h_next = []
        c_next = []
        # L3
        L3_comine = torch.cat((guided_nbr_fea_l[2], guided_ref_fea_l[2]), dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_comine))
        L3_offset_temp, c_out = self.L3_rnn(L3_offset, h_cur[0], c_cur[0])
        h_next.append(L3_offset_temp)
        c_next.append(c_out)
        L3_offset = torch.cat((L3_offset, L3_offset_temp), dim=1)
        L3_offset = self.lrelu(self.L3_rnn_conv(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack([nbr_fea_l[2], L3_offset]))

        # L2
        L2_comine = torch.cat((guided_nbr_fea_l[1], guided_ref_fea_l[1]), dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_comine))
        L2_offset_temp, c_out = self.L2_rnn(L2_offset, h_cur[1], c_cur[1])
        h_next.append(L2_offset_temp)
        c_next.append(c_out)
        L2_offset = torch.cat((L2_offset, L2_offset_temp), dim=1)
        L2_offset = self.lrelu(self.L2_rnn_conv(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))

        # L1
        L1_comine = torch.cat((guided_nbr_fea_l[0], guided_ref_fea_l[0]), dim=1)
        L1_offset = self.L1_offset_conv1(L1_comine)
        L1_offset_temp, c_out = self.L1_rnn(L1_offset, h_cur[2], c_cur[2])
        h_next.append(L1_offset_temp)
        c_next.append(c_out)
        L1_offset = torch.cat((L1_offset, L1_offset_temp), dim=1)
        L1_offset = self.lrelu(self.L1_rnn_conv(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        
        # Cascading
        offset_comine = torch.cat((L1_fea, ref_fea_l[0]), dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset_comine))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack([L1_fea, offset]))

        return L1_fea, h_next, c_next

class GCPNet(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, in_channel=1, output_channel=1, center=None):
        super(GCPNet, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.nframes = nframes
        ## GCP Branch
        self.feature_guided1 = SimpleBlock(depth=3, n_channels=nf, input_channels=in_channel*2*2, \
            output_channel=nf, kernel_size=3) # 64, H, W
        self.feature_guided1_lam = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, bias=True, padding=1)
        self.feature_guided1_beta = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, bias=True, padding=1)
        
        self.feature_guided2 = SimpleBlock(depth=3, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H, W
        self.feature_guided2_lam = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, bias=True, padding=1)
        self.feature_guided2_beta = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, bias=True, padding=1)

        self.feature_guided3 = SimpleBlock(depth=3, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H, W
        self.feature_guided3_lam = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, bias=True, padding=1)
        self.feature_guided3_beta = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, bias=True, padding=1)

        self.feature_guided4 = SimpleBlock(depth=3, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H, W
        self.feature_guided4_lam = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, bias=True, padding=1)
        self.feature_guided4_beta = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, bias=True, padding=1)

        self.feature_guided5 = SimpleBlock(depth=3, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H, W

        self.feature_guided6 = SimpleBlock(depth=3, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H, W
        self.feature_guided6_up = nn.ConvTranspose2d(in_channels=nf, out_channels=nf,\
            kernel_size=2, stride=2, padding=0, bias=True) # 64, H*2, W*2
        self.feature_guided6_lam = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, bias=True, padding=1)
        self.feature_guided6_beta = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, bias=True, padding=1)

        ## IntraF Module
        self.feature_extract = SimpleBlock(depth=5, n_channels=nf, input_channels=in_channel*4*2, \
            output_channel=nf, kernel_size=3) # 64, H, W
        self.feature_extract_acse1 = GCABlock(res_num=2, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H, W
        self.feature_extract_acse2 = GCABlock(res_num=2, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H, W
        self.feature_extract_acse3 = GCABlock(res_num=2, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H*2, W*2
        self.feature_extract_acse4 = GCABlock(res_num=2, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H*2, W*2
      
        ## InterF Module
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.merge = nn.Conv2d(nf*nframes, nf, 3, 1, 1, bias=True)

        self.feature_up = nn.ConvTranspose2d(in_channels=nf, out_channels=nf,\
            kernel_size=2, stride=2, padding=0, bias=True) # 64, H*2, W*2

        # encoder
        self.conv_block_s1 = SimpleBlock(depth=2, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H*2, W*2
        self.acse_block_s1 = DualABlock(res_num=2, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H*2, W*2
        self.pool1 = nn.Conv2d(nf, 2*nf, 3, 2, 1, bias=True) # 128 
        
        self.conv_block_s2 = SimpleBlock(depth=2, n_channels=2*nf, input_channels=2*nf, \
            output_channel=2*nf, kernel_size=3) # 128, H, W
        self.acse_block_s2 = DualABlock(res_num=2, n_channels=2*nf, input_channels=2*nf, \
            output_channel=2*nf, kernel_size=3) # 128, H, W
        self.pool2 = nn.Conv2d(2*nf, 4*nf, 3, 2, 1, bias=True) # 256

        self.conv_block_s3 = SimpleBlock(depth=2, n_channels=4*nf, input_channels=4*nf, \
            output_channel=4*nf, kernel_size=3) # 256, H//2, W//2
        self.acse_block_s3 = DualABlock(res_num=2, n_channels=4*nf, input_channels=4*nf, \
            output_channel=4*nf, kernel_size=3) # 256, H//2, W//2
        self.conv_block_s3_2 = SimpleBlock(depth=2, n_channels=4*nf, input_channels=4*nf, \
            output_channel=4*nf, kernel_size=3) # 256, H//2, W//2
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # decoder
        self.up1 = nn.ConvTranspose2d(in_channels=4*nf, out_channels=2*nf,\
            kernel_size=2, stride=2, padding=0, bias=True) # 128, H, W
        ### With SkipConnection
        # cat with conv_block_s4 # 128, H, W
        self.conv_block_s4 = SimpleBlock(depth=2, n_channels=2*nf, input_channels=4*nf, \
            output_channel=2*nf, kernel_size=3) # 128, H, W
        self.acse_block_s4 = DualABlock(res_num=2, n_channels=2*nf, input_channels=2*nf, \
            output_channel=2*nf, kernel_size=3) # 128, H, W
        
        self.up2 = nn.ConvTranspose2d(in_channels=2*nf, out_channels=nf,\
            kernel_size=2, stride=2, padding=0, bias=True) # 64, H*2, W*2
        # cat with conv_block_s3 # 64, H*2, W*2
        self.conv_block_s5 = SimpleBlock(depth=3, n_channels=nf, input_channels=2*nf, \
            output_channel=output_channel, kernel_size=3) # 64, H*2, W*2
        

    def forward(self, x, nmap):
        B, N, C, H, W = x.size()  # N video frames, C is 4 response to RGGB channel
        
        # GCP Branch
        x_gr = x[:,:,1:3,:,:].clone()
        x_gr_map = nmap[:,:,1:3,:,:].clone()
        x_gr = x_gr.view(-1, int(C/2), H, W)
        x_gr_map = x_gr_map.view(-1, int(C/2), H, W)
        temp = torch.cat([x_gr, x_gr_map], dim=1)

        x_gr1 = self.feature_guided1(temp)
        x_gr1_lam = self.feature_guided1_lam(x_gr1)
        x_gr1_beta = self.feature_guided1_beta(x_gr1)

        x_gr2 = self.feature_guided2(x_gr1)
        x_gr2_lam = self.feature_guided2_lam(x_gr2)
        x_gr2_beta = self.feature_guided2_beta(x_gr2)

        x_gr3 = self.feature_guided3(x_gr2)
        x_gr3_lam = self.feature_guided3_lam(x_gr3)
        x_gr3_beta = self.feature_guided3_beta(x_gr3)

        x_gr4 = self.feature_guided4(x_gr3)
        x_gr4_lam = self.feature_guided4_lam(x_gr4)
        x_gr4_beta = self.feature_guided4_beta(x_gr4)

        x_gr5 = self.feature_guided5(x_gr4)

        x_gr5 = x_gr5.view(B, N, -1, H, W)
        x_gr6 = self.feature_guided6(x_gr5[:, self.center, :, :, :])
        x_gr5 = x_gr5.view(B*N, -1, H, W)
        x_gr6 = self.feature_guided6_up(x_gr6)
        x_gr6_lam = self.feature_guided6_lam(x_gr6)
        x_gr6_beta = self.feature_guided6_beta(x_gr6)

        # IntraF Module
        x_temp = x.view(-1, C, H, W)
        x_nm_temp = nmap.view(-1, C, H, W)
        temp = torch.cat([x_temp, x_nm_temp], dim=1)

        x_s1 = self.feature_extract(temp)          # B*N, fea_C, H, W
        x_s1 = self.feature_extract_acse1(x_s1, x_gr1_lam, x_gr1_beta)
        x_s1 = self.feature_extract_acse2(x_s1, x_gr2_lam, x_gr2_beta)
        x_s1 = self.feature_extract_acse3(x_s1, x_gr3_lam, x_gr3_beta) # B*N, fea_C, H, W
        x_s1 = self.feature_extract_acse4(x_s1, x_gr4_lam, x_gr4_beta) # B*N, fea_C, H, W
        
        # InterF Module
        x_s1 = self.align_feature(x_s1, x_gr5, B, N, self.nf, H, W) # [B*N, fea, H, W] -> [B, N, fea, H, W]
        x_s1 = self.merge(x_s1.view(-1, self.nf*N, H, W))
        # Merge Module: encoder -- decoder
        x_s1 = self.feature_up(x_s1) # B, fea_C, H*2, W*2
        x_s1 = x_s1.mul(x_gr6_lam) + x_gr6_beta
        ###
        x_s1 = self.conv_block_s1(x_s1)       # 64, H*2, W*2
        x_s1 = self.acse_block_s1(x_s1)
        ###
        L1_temp = x_s1.clone()
        ###
        x_s2 = self.pool1(x_s1)               # 128, H, W
        x_s2 = self.conv_block_s2(x_s2)       # 128, H, W
        x_s2 = self.acse_block_s2(x_s2)       # 128, H, W
        ###
        L2_temp = x_s2.clone()
        ###
        x_s3 = self.pool2(x_s2)               # 256, H//2, W//2
        x_s3 = self.conv_block_s3(x_s3)       # 256, H//2, W//2
        x_s3 = self.acse_block_s3(x_s3)       # 256, H//2, W//2
        x_s3 = self.conv_block_s3_2(x_s3)       # 256, H//2, W//2
        
        # decoder
        out = self.up1(x_s3)                 # 128, H, W
        out = torch.cat((out, L2_temp), 1)      # 256, H, W
        out = self.conv_block_s4(out)        # 128, H, W
        out = self.acse_block_s4(out)        # 128, H, W

        out = self.up2(out)                  # 64, H*2, W*2
        out = torch.cat((out, L1_temp), 1)      # 128, H*2, W*2
        out = self.conv_block_s5(out)        # out_ch, H, W

        return out
    
    def align_feature(self, feature, guided_feature, B, N, C, H, W):
        feature_temp = torch.cat([feature, guided_feature], dim=0)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(feature_temp)) # H//2, W//2
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea)) # H//4, W//4
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = feature_temp.view(2*B, N, -1, H, W)
        L2_fea = L2_fea.view(2*B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(2*B, N, -1, H // 4, W // 4)

        #### align using DConv
        # ref feature list
        ref_fea_l = [
            L1_fea[0:B, self.center, :, :, :].clone(), L2_fea[0:B, self.center, :, :, :].clone(),
            L3_fea[0:B, self.center, :, :, :].clone()
        ]
        ref_fea_l_g = [
            L1_fea[B:, self.center, :, :, :].clone(), L2_fea[B:, self.center, :, :, :].clone(),
            L3_fea[B:, self.center, :, :, :].clone()
        ]

        aligned_fea = []
        h_cur = [None, None, None]
        c_cur = [None, None, None]
        for i in range(N):
            nbr_fea_l = [
                L1_fea[0:B, i, :, :, :].clone(), L2_fea[0:B, i, :, :, :].clone(),
                L3_fea[0:B, i, :, :, :].clone()
            ]
            nbr_fea_l_g = [
                L1_fea[B:, i, :, :, :].clone(), L2_fea[B:, i, :, :, :].clone(),
                L3_fea[B:, i, :, :, :].clone()
            ]
            a_fea, h_cur, c_cur = self.pcd_align(nbr_fea_l, ref_fea_l, nbr_fea_l_g, ref_fea_l_g, h_cur, c_cur)

            aligned_fea.append(a_fea)
            
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        return aligned_fea 




