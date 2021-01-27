# Joint Denoising and Demosaicking with Green Channel Prior for Real-world Burst Images

Implement of our GCP-Net.

Arxiv: https://arxiv.org/abs/2101.09870

## Testing
#### pretrain model: 
* store in [gcpnet_model/600000_G.pth](https://github.com/GuoShi28/GCP-Net/blob/main/experiments/gcpnet_model/600000_G.pth)
#### Testing Vid4 and REDS4:

* set data_mode in test.py to 'REDS4' and 'Vid4', the default noise level is set as the 'high noise level' mentioned in the paper.

```
python /codes/test.py
```

* To Note that: 
we only put a subset of REDS4 and Vid4 to save space, please download the full testset in official website, [RED](https://seungjunnah.github.io/Datasets/reds.html) and [Vid](http://toflow.csail.mit.edu/). More detail can refer to [data preparation](https://github.com/xinntao/EDVR/blob/master/docs/DatasetPreparation.md)

#### Testing on real captured images:
* SC_burst (Smartphone burst) Dataset: we captured 16 burst images using smartphones, and put one burst of Scene _00_ in [sub_SC_burst](https://github.com/GuoShi28/GCP-Net/tree/main/datasets/SC_burst/Scene00). We unified raw format and saved SC_burst in ".MAT", where the raw data and metadata are stored.
* Whole dataset: [BaiduYun](https://pan.baidu.com/s/1gRQ1im6Qa7vZiuOv9eO2Qw) with password d8u8.
* Bayer pattern: Our model is trained only use RGGB. Thus when testing raw images with other patterns (e.g., GRBG), don't forget to unified bayer pattern to RGGB by padding or flipping.
```
python /code/test_real.py
```

## Training
* training data preparation: Please refer to the "Video Super-Resolution" part of [data preparation](https://github.com/xinntao/EDVR/blob/master/docs/DatasetPreparation.md). To create LMDB dataset, please run [create_lmdb.py](https://github.com/GuoShi28/GCP-Net/blob/main/codes/data_scripts/create_lmdb.py).

* change training options in [train_GCP_Net.yml](https://github.com/GuoShi28/GCP-Net/blob/main/codes/options/train/train_GCP_Net.yml)
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4540 train.py -opt options/train/train_GCP_Net.yml --launcher pytorch
```
## Acknowledgement
This repo is built upon the framework of [EDVR](https://github.com/xinntao/EDVR), and we borrow some code from [Unprocessing denoising](https://github.com/timothybrooks/unprocessing), thanks for their excellent work!
