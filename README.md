# Loco3D: Indoor Multiuser Locomotion 3D Dataset
 ![demo](./assets/dataset_overview.jpg)

## Introduction

This is the official repo of our paper.

For more information, please visit our [project page](https://sites.google.com/loco3d/).

## Demo

A demo of our dataset:


## Quickstart

To setup the environment, firstly install the packages in requirements.txt:

```
pip install -r requirements.txt
```

```
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch
# [IMPORTANT] comment these two lines of code:
#   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
# [IMPORTANT] Also, you need to change l196-198 of file `[PATH-TO-VENV]/lib64/python3.8/site-packages/pointnet2_ops/pointnet2_modules.py` to `interpolated_feats = known_feats.repeat(1, 1, unknown.shape[1])`)
pip install -r requirements.txt
pip install -e .
```

Download and install [Vposer](https://github.com/nghorbani/human_body_prior), [SMPL-X](https://github.com/vchoutas/smplx)


```
bash scripts/eval.sh
```

You can download the full dataset and have a test.


### Dataset Structure & Visualization
You can refer to [README](./demo_data/README.md) for details.

### Citation
If you find this repo useful for your research, please consider citing:
