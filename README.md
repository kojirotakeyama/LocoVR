# LocoVR: Multiuser Indoor Locomotion Dataset in Virtual Reality

We present LocoVR, a dataset of two-person interactions in over 130 different indoor VR scenes, including full body pose data and highly accurate spatial information. The dataset can be used to build AI agents that operate in indoor environments, such as home robots, or to create virtual avatars for games or animations that mimic human movement and posture. 

[Evaluation code and data for the main paper is available at this link](https://anonymous.4open.science/r/LocoVR_code_test-08E6/README.md)

[Our project website is accessible from this link](https://sites.google.com/view/locovr?usp=sharing)

<!--
<center>
 <img src="./Overview.png" alt="Overview" width="600">
</center>
-->
![Overview](./Overview.png)
 - All scene data (3D geometry, semantics, textures) are derived from [Habitat-Matterport 3D Semantics Dataset (HM3DSem)](https://aihabitat.org/datasets/hm3d-semantics/) and [Habitat-Matterport 3D Dataset](https://aihabitat.org/datasets/hm3d/).

## 1. LocoVR Dataset
### 1.1 Downloading LocoVR dataset
LocoVR dataset is accessible from the following download link. 
[Download LocoVR](https://drive.google.com/drive/folders/1gE9P3MSJ6dbgpAt4YbEjZn-8cr4jtdVY?usp=drive_link)


LocoVR is provided with Python pickle files. Detailed contents of the dataset is described below.
[Contents of the dataset](./dataset_structure/README.md)

### 1.2 Quick start (visualization)
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

### 1.3 Map generation (optional)
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
  - 
## 2. LocoReal Dataset
Although LocoVR is collected in highly realistic virtual environments and is useful for learning human trajectories in relation to the surrounding environment, there remains a general concern regarding potential differences in human perception between physical and virtual spaces, which could lead to performance degradation when transferring models from virtual to real-world scenarios. To mitigate this concern, we developed LocoReal, a human trajectory dataset collected in physical space, which serves as test data to demonstrate that models trained on LocoVR can be effectively applied in real-world environments.

The real-world human trajectory data collection took place in an empty room within a campus building. Two participants performed a task in the room, where several pieces of furniture were arranged, and their 3D motions and trajectories were captured using a motion capture system. The experiment involved 5 participants across 4 different room layouts, resulting in a total of 430 collected trajectories. The data structure of LocoReal is the same as that of LocoVR.

### 2.1 Downloading LocoReal dataset
LocoReal dataset is accessible from the following download link. 
[Download LocoReal](https://drive.google.com/drive/folders/1C7VANAopABgg_NgfvWAryb5NmcBgawbL?usp=sharing)

### 2.2 Quick start (visualization)
1. Go to the "LocoReal" folder downloaded from the above link.
2. Install the packages in requirements.txt (python==3.8.1, cuda12.1):
```
pip install -r requirements.txt
```
3. Visualize the trajectories
   If you conduct vis_trajectory.py, you will get time-sereis trajectory images of trajectories in 4 scenes contained in the LocoReal.
```
python ./vis_traj.py
```
## 3 Test Code
Here we provide the test code of global path prediction. This task estimates a static global path from a starting point to a goal location, which can be used to predict human global paths or plan human-like global paths for robots. Our dataset demonstrates the ability to learn such human-like paths that consider obstacle avoidance, efficiency, and social motion behaviors, such as maintaining social distance when passing or choosing longer routes to avoid collisions. The input includes the past trajectories of two people, p1 and p2 (length=1.0s, interval=0.067s), past heading directions of p1 and p2 (length=1.0s, interval=0.067s), scene map, and goal position. The output is a static path from the start to the goal.
Here we provide the model trained by LocoVR, test code and the input data (LocoReal) to evaluate the model.

### 3.1 Downloading the code and input data
The test code with the input data are accessible from the following download link.
[Download TestCode](https://drive.google.com/drive/folders/10ILf7YTiznbzh5pc8CiHkP3Cvlz5Kt_0?usp=sharing)

### 3.2 Quick start (Global path prediction and visualization)
1. Go to the "GlobalPathPrediction_testcode" folder downloaded from the above link.
2. Install the packages in requirements.txt (python==3.8.1, cuda12.1):
```
pip install -r requirements.txt
```
3. Run the code to predict global path and visualize the result
   If you conduct run.py, the model predicts global path and you will get time-series images of predicted global path (red) and groundtruth path (green)
```
python ./run.py
```

## Evaluation codes for the main paper
Evaluation code and data for the main paper is available at this [link](https://anonymous.4open.science/r/LocoVR_code_test-08E6/README.md)

## Citation
If you find this repo useful for your research, please consider citing:

## License
This project is licensed under the MIT License
