# Dataset structure

Following figure shows the structure of LocoVR. The data for each scene is stored in a Python pickle file. The scene number corresponds to those in the scene names of the Habitat-Matterport 3D dataset (HM3D). Each pickle file contains multiple trajectory data in a Python list format; data[0] to data[n]. Each element in the list corresponds to a trajectory sequence from the start to the goal, including motion data for two persons (p1 and p2), the scene ID, the time label of the motion data, and the goal position of p1.

The motion data for p1 and p2 includes full body motion data: head, waist, right hand, left hand, right foot, and left foot. Each body part has a 3D position and a 4D quaternion in the world coordinate system of HM3D. See the sample code on the main page on the repo to learn how HM3D and LocoVR are aligned to the same coordinate system. 

Futher deitals of the variables are described below.

**['scene_id']**
This corresponds to the number in the scene names of HM3D, which is also the name of the pickle file.

**['time']**
This is a numpy array of time labels corresponding to the time-series motion data. The starting time is 0, and the final time label indicates when p1 arrives at the goal.

**['goal']**
This contains the 3D position of the goal in a numpy array, using the world coordinate system of HM3D.

**['head'], ['waist'], ['right hand'], ['left hand'], ['right foot'], ['left foot']**
These contain position and pose data captured by the HTC VIVE motion tracker system. "pos" includes 3D position data, and "pose" includes 4D quaternion data. Both are represented in the world coordinate system of HM3D, using a numpy array format.

<center>
 <img src="../assets/Dataset structure.png" alt="structure" width="800">
</center>
