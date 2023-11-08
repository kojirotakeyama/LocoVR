import numpy as np
import os
import pickle
import copy
import cv2
import yaml
from scipy.spatial.transform import Rotation as R

class PosPose:

    def __init__(self):
        self.pos = np.zeros([1,3])
        self.pose = np.zeros([1,4])
        
class Body:

    def __init__(self):
        
        self.head = PosPose()
        self.hand_l = PosPose()
        self.hand_r = PosPose()
        self.waist = PosPose()
        self.foot_l = PosPose()
        self.foot_r = PosPose()
        
def quat2euler(q):
    
    r = R.from_quat(q.tolist())
    return r.as_euler('zxy')

def euler2quat(euler):
    
    r = R.from_euler('zxy', euler.tolist())
    return r.as_quat()

def rotquat(q,rot):
    
    euler = quat2euler(q)
    euler += rot
    q2 = euler2quat(euler)
    
    return q2

def conv_meter2pixel(pos_m, center):
    
    if len(pos_m.shape)==2:
        pos_x_px = ((pos_m[:,0]-center[0]+size_m/2)*size_px/size_m).astype(np.int16)
        pos_y_px = ((pos_m[:,1]-center[1]+size_m/2)*size_px/size_m).astype(np.int16)
        pos_out = np.concatenate([pos_x_px.reshape(-1,1),pos_y_px.reshape(-1,1)],axis=1)
    else:
        pos_x_px = ((pos_m[0]-center[0]+size_m/2)*size_px/size_m).astype(np.int16)
        pos_y_px = ((pos_m[1]-center[1]+size_m/2)*size_px/size_m).astype(np.int16)
        pos_out = np.array([pos_x_px,pos_y_px])
    
    return pos_out

def get_center(path, scene_id):
    
    data = np.genfromtxt(path ,delimiter=',')
    
    c1 = np.zeros(3)
    c2 = np.zeros(3)
    
    if data[:,0].tolist().count(scene_id)>0:
        idx = data[:,0].tolist().index(scene_id)
        c1 = data[idx,1:4]
        c2 = data[idx,4:7]
    
    return np.array([(c1[0]+c2[0])/2,(c1[1]+c2[1])/2])

def transform_axis_pos(val_in):
    
    val_out = np.zeros(val_in.shape)
    
    val_out[:,0] = val_in[:,2]
    val_out[:,1] = -val_in[:,0]
    val_out[:,2] = val_in[:,1]
        
    return val_out

def transform_axis_pose(q):
    
    if q.shape[0] == 0:
        return np.zeros(q.shape)
    
    q2 = copy.deepcopy(q)  #left hand to right hand coordinate
    q2[:,1] = -q2[:,1]
    q2[:,3] = -q2[:,3]
    q2 = rotquat(q2,np.array([0,0,np.pi]))
        
    return q2

def generate_anim(p1, p2, g1, scene_map, traj_id):
    
    scene_map_tmp1 = copy.deepcopy(scene_map)
    scene_map_tmp2 = copy.deepcopy(scene_map)
    
    cv2.circle(scene_map_tmp1, (g1[0], g1[1]), 20, (0, 0, 255), thickness=-1)
    cv2.circle(scene_map_tmp2, (g1[0], g1[1]), 20, (0, 0, 255), thickness=-1)
    
    if not os.path.exists('./viz_traj/scene_{:0=3}'.format(scene_id)):
        os.makedirs('./viz_traj/scene_{:0=3}'.format(scene_id))
        
    for i in range(1,p1.waist.pos.shape[0]):
        
        #P1
        x1,y1 = p1.waist.pos[i-1][0],p1.waist.pos[i-1][1]
        x2,y2 = p1.waist.pos[i][0],p1.waist.pos[i][1]
        yaw = -quat2euler(p1.waist.pose[i])[2]

        if x1<0 or y1<0 or x2<0 or y2<0 or x1>size_px or y1>size_px or x2>size_px or y2>size_px:
            continue
            
        cv2.line(scene_map_tmp1, (x1, y1), (x2, y2), (0, 0, 255), thickness=3, lineType=cv2.LINE_4)
        
        if np.mod(i,10)==0 or i==len(p1.waist.pos)-1:
            
            x1 = int(x2 - 30*np.cos(yaw))
            y1 = int(y2 - 30*np.sin(yaw))
            
            cv2.arrowedLine(scene_map_tmp2, (x1, y1), (x2, y2), (0, 0, 255), thickness=10, tipLength=0.5)

        #P2
        x1,y1 = p2.waist.pos[i-1][0],p2.waist.pos[i-1][1]
        x2,y2 = p2.waist.pos[i][0],p2.waist.pos[i][1]
        yaw = -quat2euler(p2.waist.pose[i])[2]

        if x1<0 or y1<0 or x2<0 or y2<0 or x1>size_px or y1>size_px or x2>size_px or y2>size_px:
            continue
                
        cv2.line(scene_map_tmp1, (x1, y1), (x2, y2), (255, 0, 0), thickness=3, lineType=cv2.LINE_4)
            
        if np.mod(i,10)==0 or i==p1.waist.pos.shape[0]-1:
                
            x1 = int(x2 - 30*np.cos(yaw))
            y1 = int(y2 - 30*np.sin(yaw))
                
            cv2.arrowedLine(scene_map_tmp2, (x1, y1), (x2, y2), (255, 0, 0), thickness=10, tipLength=0.5)

            cv2.imwrite('./viz_traj/scene_{:0=3}'.format(scene_id) + '/traj_{:0=3}'.format(exp_id) + '_{:0=3}'.format(traj_id) + '_{:0=3}'.format(i) + '.png', scene_map_tmp2)
                        
        scene_map_tmp2 = copy.deepcopy(scene_map_tmp1)

def generate_anim2(p1, p2, g1, scene_map, traj_id):
    
    scene_map_tmp1 = copy.deepcopy(scene_map)
    scene_map_tmp2 = copy.deepcopy(scene_map)
    
    cv2.circle(scene_map_tmp1, (g1[0], g1[1]), 20, (0, 0, 255), thickness=-1)
    cv2.circle(scene_map_tmp2, (g1[0], g1[1]), 20, (0, 0, 255), thickness=-1)
    
    if not os.path.exists('./viz_traj/scene_{:0=3}'.format(scene_id)):
        os.makedirs('./viz_traj/scene_{:0=3}'.format(scene_id))
        
    for i in range(1,p1.waist.pos.shape[0]):
        
        #P1
        x1,y1 = p1.waist.pos[i-1][0],p1.waist.pos[i-1][1]
        x2,y2 = p1.waist.pos[i][0],p1.waist.pos[i][1]
        #yaw = quat2euler(p1.waist.pose[i])[2]

        if x1<0 or y1<0 or x2<0 or y2<0 or x1>size_px or y1>size_px or x2>size_px or y2>size_px:
            continue
            
        cv2.line(scene_map_tmp1, (x1, y1), (x2, y2), (0, 0, 255), thickness=3, lineType=cv2.LINE_4)
        
        if np.mod(i,10)==0 or i==len(p1.waist.pos)-1:
            
            waist = p1.waist.pos[i]
            head = p1.head.pos[i]
            hand_l = p1.hand_l.pos[i]
            hand_r = p1.hand_r.pos[i]
            foot_l = p1.foot_l.pos[i]
            foot_r = p1.foot_r.pos[i]
            
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(foot_l), (255, 0, 255), thickness=3, lineType=cv2.LINE_4)
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(foot_r), (255, 0, 255), thickness=3, lineType=cv2.LINE_4)
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(hand_l), (0, 255, 0), thickness=3, lineType=cv2.LINE_4)
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(hand_r), (0, 255, 0), thickness=3, lineType=cv2.LINE_4)

        #P2
        x1,y1 = p2.waist.pos[i-1][0],p2.waist.pos[i-1][1]
        x2,y2 = p2.waist.pos[i][0],p2.waist.pos[i][1]
        #yaw = quat2euler(p2.waist.pose[i])[2]

        if x1<0 or y1<0 or x2<0 or y2<0 or x1>size_px or y1>size_px or x2>size_px or y2>size_px:
            continue
                
        cv2.line(scene_map_tmp1, (x1, y1), (x2, y2), (255, 0, 0), thickness=3, lineType=cv2.LINE_4)
            
        if np.mod(i,10)==0 or i==p1.waist.pos.shape[0]-1:
                
            waist = p2.waist.pos[i]
            head = p2.head.pos[i]
            hand_l = p2.hand_l.pos[i]
            hand_r = p2.hand_r.pos[i]
            foot_l = p2.foot_l.pos[i]
            foot_r = p2.foot_r.pos[i]
                
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(foot_l), (255, 0, 255), thickness=3, lineType=cv2.LINE_4)
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(foot_r), (255, 0, 255), thickness=3, lineType=cv2.LINE_4)
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(hand_l), (0, 255, 0), thickness=3, lineType=cv2.LINE_4)
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(hand_r), (0, 255, 0), thickness=3, lineType=cv2.LINE_4)

            cv2.imwrite('./viz_traj/scene_{:0=3}'.format(scene_id) + '/traj_{:0=3}'.format(exp_id) + '_{:0=3}'.format(traj_id) + '_{:0=3}'.format(i) + '.png', scene_map_tmp2)
                        
        scene_map_tmp2 = copy.deepcopy(scene_map_tmp1)

def get_pospose(traj):
    
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
    
    p = Body()
    p.head.pos = transform_axis_pos(traj[:,1:4])
    p.head.pose = transform_axis_pose(traj[:,4:8])
    p.hand_l.pos = transform_axis_pos(traj[:,8:11])
    p.hand_l.pose = transform_axis_pose(traj[:,11:15])
    p.hand_r.pos = transform_axis_pos(traj[:,15:18])
    p.hand_r.pose = transform_axis_pose(traj[:,18:22])
    p.waist.pos = transform_axis_pos(traj[:,22:25])
    p.waist.pose = transform_axis_pose(traj[:,25:29])
    p.foot_l.pos = transform_axis_pos(traj[:,29:32])
    p.foot_l.pose = transform_axis_pose(traj[:,32:36])
    p.foot_r.pos = transform_axis_pos(traj[:,36:39])
    p.foot_r.pose = transform_axis_pose(traj[:,39:43])
    
    return p

def conv_scale(p_in, img_center):
    
    p_out = copy.deepcopy(p_in)
    p_out.head.pos = conv_meter2pixel(p_in.head.pos, img_center)
    p_out.hand_l.pos = conv_meter2pixel(p_in.hand_l.pos, img_center)
    p_out.hand_r.pos = conv_meter2pixel(p_in.hand_r.pos, img_center)
    p_out.waist.pos = conv_meter2pixel(p_in.waist.pos, img_center)
    p_out.foot_r.pos = conv_meter2pixel(p_in.foot_l.pos, img_center)
    p_out.foot_l.pos = conv_meter2pixel(p_in.foot_r.pos, img_center)
    
    return p_out
    
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
        
        #convert unity coordinate to world coordinate
        p1 = get_pospose(traj1[i])
        p2 = get_pospose(traj2[i])
        g1 = transform_axis_pos(goal1[i].reshape(1,-1))
        
        #convert world coordinate[m] to image coordinate[pixel]
        p1 = conv_scale(p1, img_center)
        p2 = conv_scale(p2, img_center)
        g1 = conv_meter2pixel(g1[0,:2], img_center)
        
        #Generate and save time-series trajectory image
        if type_viz == 'waist':
            generate_anim(p1, p2, g1, scene_map, i)
        if type_viz == 'body':
            generate_anim2(p1, p2, g1, scene_map, i)

if __name__ == "__main__":
    
    with open('./config.yaml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        
    path_pickle = params['path_pickle']
    path_map = params['path_map']
    path_center = params['path_center']
    
    size_m = params['size_m']
    size_px = params['size_px']
    
    scene_id_list = params['scene_id']
    exp_id = params['exp_id']
    
    type_viz = params['type_viz']

    for scene_id in scene_id_list:
        disp_trajectory()