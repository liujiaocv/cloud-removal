## Installation

```
conda create -n cmnet
conda activate cmnet
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm

Install warmup scheduler
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

## Training

```
python train.py
```

## Evaluation

```
python test_cloud.py
```

#### To reproduce PSNR/SSIM scores of the paper, run

```
evaluate_PSNR_SSIM.m 
```

## Citation

If you use CMNet, please consider citing:

@ARTICLE{10466744,
  author={Liu, Jiao and Pan, Bin and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Cascaded Memory Network for Optical Remote Sensing Imagery Cloud Removal}, 
  year={2024}
}

## Acknowledgement

This project is mainly based on https://github.com/swz30/MPRNet
