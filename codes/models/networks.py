import torch
import models.archs.arch_gcpnet as arch_gcpnet

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    # burst restoration
    if which_model == 'GCPNet':
        netG = arch_gcpnet.GCPNet(nf=opt_net['nf'],\
                groups=opt_net['groups'], in_channel=1, output_channel=3)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
