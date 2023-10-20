import numpy as np
import os
import pickle
import copy
import cv2
import yaml
from scipy.spatial.transform import Rotation as R

def quat2euler(q):
    
    r = R.from_quat(q.tolist())
    return r.as_euler('zxy')

def euler2quat(euler):
    
    r = R.from_euler('zxy', euler.tolist())
    return r.as_quat()

def conv_meter2pixel(pos_m, center):
    
    if len(pos_m.shape)==2:
        pos_px = ((pos_m-center.reshape(1,-1)+size_m/2)*size_px/size_m).astype(np.int16)
    else:
        pos_px = ((pos_m-center+size_m/2)*size_px/size_m).astype(np.int16)
    
    return pos_px

def get_center(path, scene_id):
    
    data = np.genfromtxt(path ,delimiter=',')
    
    c1 = np.zeros(3)
    c2 = np.zeros(3)
    
    if data[:,0].tolist().count(scene_id)>0:
        idx = data[:,0].tolist().index(scene_id)
        c1 = data[idx,1:4]
        c2 = data[idx,4:7]
    
    return np.array([(c1[1]+c2[1])/2,(c1[0]+c2[0])/2])

def generate_anim(pos1, pos2, pose1, pose2, g1, scene_map, traj_id):
    
    scene_map_tmp1 = copy.deepcopy(scene_map)
    scene_map_tmp2 = copy.deepcopy(scene_map)
    
    cv2.circle(scene_map_tmp1, (g1[0], g1[1]), 20, (0, 0, 255), thickness=-1)
    cv2.circle(scene_map_tmp2, (g1[0], g1[1]), 20, (0, 0, 255), thickness=-1)
    
    if not os.path.exists('./viz_traj/scene_{:0=3}'.format(scene_id)):
        os.makedirs('./viz_traj/scene_{:0=3}'.format(scene_id))
        
    for i in range(1,pos1.shape[0]):
        
        #P1
        x1,y1 = pos1[i-1][0],pos1[i-1][1]
        x2,y2 = pos1[i][0],pos1[i][1]
        yaw = quat2euler(pose1[i])[2]

        if x1<0 or y1<0 or x2<0 or y2<0 or x1>size_px or y1>size_px or x2>size_px or y2>size_px:
            continue
            
        cv2.line(scene_map_tmp1, (x1, y1), (x2, y2), (0, 0, 255), thickness=3, lineType=cv2.LINE_4)
        
        if np.mod(i,10)==0 or i==len(pos1)-1:
            
            x1 = int(x2 - 30*np.cos(yaw))
            y1 = int(y2 - 30*np.sin(yaw))
            
            cv2.arrowedLine(scene_map_tmp2, (x1, y1), (x2, y2), (0, 0, 255), thickness=10, tipLength=0.5)

        #P2
        x1,y1 = pos2[i-1][0],pos2[i-1][1]
        x2,y2 = pos2[i][0],pos2[i][1]
        yaw = quat2euler(pose2[i])[2]

        if x1<0 or y1<0 or x2<0 or y2<0 or x1>size_px or y1>size_px or x2>size_px or y2>size_px:
            continue
                
        cv2.line(scene_map_tmp1, (x1, y1), (x2, y2), (255, 0, 0), thickness=3, lineType=cv2.LINE_4)
            
        if np.mod(i,10)==0 or i==pos1.shape[0]-1:
                
            x1 = int(x2 - 30*np.cos(yaw))
            y1 = int(y2 - 30*np.sin(yaw))
                
            cv2.arrowedLine(scene_map_tmp2, (x1, y1), (x2, y2), (255, 0, 0), thickness=10, tipLength=0.5)

            cv2.imwrite('./viz_traj/scene_{:0=3}'.format(scene_id) + '/traj_{:0=3}'.format(exp_id) + '_{:0=3}'.format(traj_id) + '_{:0=3}'.format(i) + '.png', scene_map_tmp2)
                        
        scene_map_tmp2 = copy.deepcopy(scene_map_tmp1)

def disp_trajectory():
    
    #create output folder
    if not os.path.exists('./viz_traj'):
        os.makedirs('./viz_traj')
    
    #load data from pickle file
    f = open(path_pickle + '/{:0=3}'.format(scene_id),"rb")
    data = pickle.load(f)
    
    #load scene maps
    scene_map = cv2.imread(path_map + '/{:0=3}.png'.format(scene_id))
    
    #load image center point in the world coordinate[m]
    img_center = get_center(path_center, scene_id)
    
    #extract trajectory data. traj1 and traj2 are trajectories for p1 and p2, and g1 is goal for p1.
    traj1 = data[exp_id][0]
    traj2 = data[exp_id][1]
    goal1 = data[exp_id][2]
    
    #number of trajectories in the experiment
    num_traj = len(traj1)
    
    for i in range(num_traj):
        
        #convert world coordinate[m] to image coordinate[pixel]
        #traj1[N][T,M]: 
        #   N: number of trajectories in the experiment
        #   T: length (=time steps) of trajectory
        #   M: Data index
        #       M=0: time
        #       M=1-3: 3d position of head[m]           M=4-7: 3d pose of head [quaternion]
        #       M=8-10: 3d position of left hand[m]     M=11-14: 3d pose of left hand [quaternion]        
        #       M=15-17: 3d position of right hand[m]   M=18-21: 3d pose of right hand [quaternion]
        #       M=22-24: 3d position of waist[m]        M=25-28: 3d pose of waist [quaternion]
        #       M=29-31: 3d position of left foot[m]    M=32-25: 3d pose of left foot [quaternion]
        #       M=36-38: 3d position of right foot[m]   M=39-42: 3d pose of right foot [quaternion]
        
        #traj1[i][:,23] = -traj1[i][:,23]
        #traj2[i][:,23] = -traj2[i][:,23]
        
        pos1 = conv_meter2pixel(traj1[i][:,22:24], img_center)
        pos2 = conv_meter2pixel(traj2[i][:,22:24], img_center)
        pose1 = traj1[i][:,25:29]
        pose2 = traj2[i][:,25:29]
        g1 = conv_meter2pixel(goal1[i][0:2], img_center)
        
        
        generate_anim(pos1, pos2, pose1, pose2, g1, scene_map, i)

if __name__ == "__main__":
    
    with open('./config.yaml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        
    path_pickle = params['path_pickle']
    path_map = params['path_map']
    path_center = params['path_center']
    
    size_m = params['size_m']
    size_px = params['size_px']
    
    scene_id = params['scene_id']
    exp_id = params['exp_id']

    disp_trajectory()