# LocoVR: Multiuser Indoor Locomotion Dataset in Virtual Reality

We present LocoVR, a dataset of two-person interactions in over 130 different indoor VR scenes, including full body pose data and highly accurate spatial information. The dataset can be used to build AI agents that operate in indoor environments, such as home robots, or to create virtual avatars for games or animations that mimic human movement and posture. 

[Evaluation code and data for ICLR2025 is available at this link](https://anonymous.4open.science/r/LocoVR_code_test-08E6/README.md)

[Our project website is accessible from this link](https://anonymous.4open.science/r/LocoVR-1B87/README.md)

<!--
<center>
 <img src="./Overview.png" alt="Overview" width="600">
</center>
-->
![Overview](./Overview.png)
 - All scene data (3D geometry, semantics, textures) are derived from [Habitat-Matterport 3D Semantics Dataset (HM3DSem)](https://aihabitat.org/datasets/hm3d-semantics/) and [Habitat-Matterport 3D Dataset](https://aihabitat.org/datasets/hm3d/).

## Downloading LocoVR dataset
LocoVR dataset is accessible from the following download link. 
[Download LocoVR](https://drive.google.com/drive/folders/1gE9P3MSJ6dbgpAt4YbEjZn-8cr4jtdVY?usp=drive_link)


LocoVR is provided with Python pickle files. Detailed contents of the dataset is described below.
[Contents of the dataset](./dataset_structure/README.md)

## Quick start
Quick visualization of the trajectories contained in LocoVR is done by the following instructions.

1. Download LocoVR dataset
   Download the folder "LocoVR" from following link and unzip it, then place it in the top of "visualize_trajectory" folder.
   [Download LocoVR](https://drive.google.com/drive/folders/1gE9P3MSJ6dbgpAt4YbEjZn-8cr4jtdVY?usp=drive_link)

3. Download map images
   Download the foloder "Maps" from following link and unzip it, then place it in the top of "visualize_trajectory" folder.
   [Download map images](https://drive.google.com/drive/folders/1bUT8aHKJmPwvhUFINHDCNmgfyR1vT33G?usp=sharing)
     
4. Install the packages in requirements.txt (python==3.8.1, cuda12.1):
```
pip install -r requirements.txt
```
4. Visualize the trajectory
   If you conduct vis_trajectory.py, you will get time-sereis trajectory images of the specified scenes.
```
python ./visualize_trajectory/vis_traj.py
```
Tips: 
- To change the scene of visualization, edit "scene_id" in the config.yaml. You can choose multiple scenes with a list style.
- To change the map type, edit "map type" in the config.yaml.
- To change the type of visualizing trajectory, edit "type_viz" to "waist" or "body".

## Map generation
All the 2D maps provided in the "Quick start" are generated basd on HM3DSem datasets.
Following instruction will help you generating the maps.

1. Download Habitat Dataset
Download Habitat Dataset from the following link and place it on the directory you specified on the config.yaml.
To generate the maps, download HM3DSem datasets (hm3d-train-semantic-annots-v0.2.tar/hm3d-val-semantic-annots-v0.2.tar) from the following link.
- [HM3DSem](https://github.com/matterport/habitat-matterport-3dresearch/tree/main)

If you just want to test the map generation code, you can do it with small sample data: hm3d-example-semantic-annots-v0.2.tar

2. Generate maps
  You can generate 2D maps from the HM3DSem Dataset through running generate_map.py as follows.
```
python ./Generate_map/generate_map.py
```
Tips: 
- To change the map type (binary, height, semantic, texture), modify "map_type" in the config.yaml.
- If you need to generate photo realistic texture map, download HM3D (including .obj) from the following link.
  - [HM3D](https://matterport.com/partners/facebook)

## Evaluation codes for ICLR2025
Evaluation code and data for ICLR2025 is available at [ICLR2025](https://anonymous.4open.science/r/LocoVR_code_test-08E6/README.md)

## Citation
If you find this repo useful for your research, please consider citing:

## License
This project is licensed under the MIT License
