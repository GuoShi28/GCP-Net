# Joint Denoising and Demosaicking with Green Channel Prior for Real-world Burst Images

Implement of our GCP-Net.

Arxiv: https://arxiv.org/abs/2101.09870

## Testing
* pretrain model: store in [gcpnet_model/600000_G.pth](https://github.com/GuoShi28/GCP-Net/blob/main/experiments/gcpnet_model/600000_G.pth)
* Testing Vid4 and REDS4:

set data_mode in test.py to 'REDS4' and 'Vid4', the default noise level is set as the 'high noise level' mentioned in the paper.

```
python /codes/test.py
```
* Testing on real captured images:

TBA (We test real-world data using the same model as REDS4 and Vid4, the testset is still preparing.)

## Training
* training data preparation: Please refer to the "Video Super-Resolution" part of [data preparation](https://github.com/xinntao/EDVR/blob/master/docs/DatasetPreparation.md). To create LMDB dataset, please run [create_lmdb.py](https://github.com/GuoShi28/GCP-Net/blob/main/codes/data_scripts/create_lmdb.py).

* change training options in [train_GCP_Net.yml](https://github.com/GuoShi28/GCP-Net/blob/main/codes/options/train/train_GCP_Net.yml)
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4540 train.py -opt options/train/train_GCP_Net.yml --launcher pytorch
```
## Acknowledgement
This repo is built upon the framework of [EDVR](https://github.com/xinntao/EDVR), and we borrow some code from [Unprocessing denoising](https://github.com/timothybrooks/unprocessing), thanks for their excellent work!
