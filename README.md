# LocoVR: Multiuser Indoor Locomotion Dataset

We present LocoVR, a dataset of multi-person interactions in over 130 different indoor VR scenes, including 3D body pose data and highly accurate spatial information. The dataset can be used to build AI agents that operate in indoor environments, such as home robots, or to create virtual avatars for games or animations that mimic human movement and posture. 
For more information, please visit our [project page](https://sites.google.com/view/loco3d/home).

<center>
 <img src="./assets/Overview.png" alt="demo" width="800">
</center>

 - Blue and red curves depict two people's trajectories for one data collection session.
 - All scene data (3D geometry, semantics, textures) are derived from [Habitat-Matterport 3D Semantics Dataset (HM3DSem)](https://aihabitat.org/datasets/hm3d-semantics/) and [Habitat-Matterport 3D Dataset](https://aihabitat.org/datasets/hm3d/).

## Downloading LocoVR dataset
Loco3D dataset can be downloaded from the following link. 

[Download LocoVR](https://drive.google.com/drive/folders/1wU82rCurMfYkCV7TvSPcSVslW79Mj7Pp?usp=drive_link)

Structure of the dataset is described below.

[Dataset contents](./dataset_contents/README.md)

## Quick start
Visualizing trajectories in 2D plane can be done by following instructions.

1. Download LocoVR dataset
   Download the folder "LocoVR" from following link and unzip it, then place it in the top of "visualize_trajectory" folder.
   [Download LocoVR](https://drive.google.com/drive/folders/1wU82rCurMfYkCV7TvSPcSVslW79Mj7Pp?usp=drive_link)

3. Download map images
   Download the foloder "Maps" from following link and unzip it, then place it in the top of "visualize_trajectory" folder.
   [Download map images](https://drive.google.com/drive/folders/1g4QO8WV4hfHFgCzfBLTi42zCJ6km7JoH?usp=drive_link)
     
4. Install the packages in requirements.txt:
```
pip install -r requirements.txt
```
4. Visualize the trajectory
   If you conduct vis_trajectory.py, you will get time-sereis trajectory images of the specified scenes.
```
python ./visualize_trajectory/vis_traj.py
```
Tip: 
- To change the scene of visualization, edit "scene_id" and "exp_id" in the config.yaml.
- To change the map type, edit "map type" in the config.yaml.
- To change the type of visualizing trajectory, edit "type_viz" to "waist" or "body".

## Map generation
All the 2D maps used in the "Quick start" are generated from HM3DSem Dataset.
Following instruction will help you generating maps.

1. Download Habitat Dataset
Download Habitat Dataset from the following link and place it on the "Generate_map" folder.
To generate all the maps for trajectories included in Loco3D dataset, you need to download following datasets.
- [HM3DSem Train](https://api.matterport.com/resources/habitat/hm3d-train-semantic-annots-v0.2.tar)
- [HM3DSem Val](https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.2.tar)

If you just want to test the code, you can do it with small sample data.
- [HM3DSem Example](https://github.com/matterport/habitat-matterport-3dresearch/blob/main/example/hm3d-example-semantic-annots-v0.2.tar)

2. Generate maps
  You can generate 2D maps from the Habitat Dataset conducting the code below.
```
python ./Generate_map/generate_map.py
```
Tip: 
- To change the map type (binary, height, semantic), modify "map_type" in the config.yaml.
- If you need to generate photo realistic textured map, download HM3D (including .obj) from the following link.
  - [HM3D](https://matterport.com/partners/facebook)

## Citation
If you find this repo useful for your research, please consider citing:
