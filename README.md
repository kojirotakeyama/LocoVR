# Loco3D: Indoor Multiuser Locomotion 3D Dataset

We present Loco3D, a dataset of multi-person interactions in over 100 different indoor VR scenes, including 3D body pose data and highly accurate spatial information. The dataset can be used to build AI agents that operate in indoor environments, such as home robots, or to create virtual avatars for games or animations that mimic human movement and posture. 
For more information, please visit our [project page](https://sites.google.com/loco3d/).

 ![demo](./assets/scenes_in_loco3d_v3.png)

 - Blue and red curves depict two people's trajectories for one data collection session.
 - All the scene data is derived from Habitat-Matterport [3D Semantics Dataset (HM3DSem)](https://aihabitat.org/datasets/hm3d-semantics/) and [Habitat-Matterport 3D Dataset](https://aihabitat.org/datasets/hm3d/).

## Demo Video

A demo of our dataset:

## Downloading Loco3D dataset
Loco3D dataset can be downloaded from the following link.

[Download Loco3D](https://).

Structure of the dataset is described below.

[Dataset structure](./dataset_structure/README.md)

## Quick start
Visualizing trajectories in 2D plane can be done by following instructions.

1. Download Loco3D dataset
   Download and unzip the dataset, then place it on the top directory of the git folder.
   
2. Install the packages in requirements.txt:
```
pip install -r requirements.txt
```
3. Visualize the trajectory
   If you conduct vis_trajectory.py, you will get time-sereis trajectory images for the specified scene and experiment number.
```
python vis_trajectory.py
```
TIP: 
To change the scene of visualization, edit "scene_id" and "exp_id" in the config.yaml.
To change the map type, edit "map type" in the config.yaml. 

## Map generation
All the 2D maps used in the "Quick start" are generated from Habitat Dataset.
You can test generating maps following the instruction below.

1. Download Habitat Dataset
Download Habitat Dataset from the following link and place it on the top directory of the git folder.
You will need to sign up your account for downloading the dataset.
[Downloading Habitat Dataset](https://)

2. Generate maps
You can generate 2D maps from the Habitat Dataset conducting the code below.
```
python generate_map.py
```

## Citation
If you find this repo useful for your research, please consider citing:
