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
        
    for i in range(1,p1['waist']['pos'].shape[0]):
        
        #P1
        x1,y1 = p1['waist']['pos'][i-1][0],p1['waist']['pos'][i-1][1]
        x2,y2 = p1['waist']['pos'][i][0],p1['waist']['pos'][i][1]
        yaw = -quat2euler(p1['waist']['pose'][i])[2]

        if x1<0 or y1<0 or x2<0 or y2<0 or x1>size_px or y1>size_px or x2>size_px or y2>size_px:
            continue
            
        cv2.line(scene_map_tmp1, (x1, y1), (x2, y2), (0, 0, 255), thickness=3, lineType=cv2.LINE_4)
        
        if np.mod(i,10)==0 or i==len(p1['waist']['pos'])-1:
            
            x1 = int(x2 - 30*np.cos(yaw))
            y1 = int(y2 - 30*np.sin(yaw))
            
            cv2.arrowedLine(scene_map_tmp2, (x1, y1), (x2, y2), (0, 0, 255), thickness=10, tipLength=0.5)

        #P2
        x1,y1 = p2['waist']['pos'][i-1][0],p2['waist']['pos'][i-1][1]
        x2,y2 = p2['waist']['pos'][i][0],p2['waist']['pos'][i][1]
        yaw = -quat2euler(p2['waist']['pose'][i])[2]

        if x1<0 or y1<0 or x2<0 or y2<0 or x1>size_px or y1>size_px or x2>size_px or y2>size_px:
            continue
                
        cv2.line(scene_map_tmp1, (x1, y1), (x2, y2), (255, 0, 0), thickness=3, lineType=cv2.LINE_4)
            
        if np.mod(i,10)==0 or i==p1['waist']['pos'].shape[0]-1:
                
            x1 = int(x2 - 30*np.cos(yaw))
            y1 = int(y2 - 30*np.sin(yaw))
                
            cv2.arrowedLine(scene_map_tmp2, (x1, y1), (x2, y2), (255, 0, 0), thickness=10, tipLength=0.5)

            cv2.imwrite('./viz_traj/scene_{:0=3}'.format(scene_id) + '/{:0=3}'.format(traj_id) + '_{:0=3}'.format(i) + '.png', scene_map_tmp2)
                        
        scene_map_tmp2 = copy.deepcopy(scene_map_tmp1)

def generate_anim2(p1, p2, g1, scene_map, traj_id):
    
    scene_map_tmp1 = copy.deepcopy(scene_map)
    scene_map_tmp2 = copy.deepcopy(scene_map)
    
    cv2.circle(scene_map_tmp1, (g1[0], g1[1]), 20, (0, 0, 255), thickness=-1)
    cv2.circle(scene_map_tmp2, (g1[0], g1[1]), 20, (0, 0, 255), thickness=-1)
    
    if not os.path.exists('./viz_traj/scene_{:0=3}'.format(scene_id)):
        os.makedirs('./viz_traj/scene_{:0=3}'.format(scene_id))
        
    for i in range(1,p1['waist']['pos'].shape[0]):
        
        #P1
        x1,y1 = p1['waist']['pos'][i-1][0],p1['waist']['pos'][i-1][1]
        x2,y2 = p1['waist']['pos'][i][0],p1['waist']['pos'][i][1]
        #yaw = quat2euler(p1.waist.pose[i])[2]

        if x1<0 or y1<0 or x2<0 or y2<0 or x1>size_px or y1>size_px or x2>size_px or y2>size_px:
            continue
            
        cv2.line(scene_map_tmp1, (x1, y1), (x2, y2), (0, 0, 255), thickness=3, lineType=cv2.LINE_4)
        
        if np.mod(i,10)==0 or i==len(p1['waist']['pos'])-1:
            
            waist = p1['waist']['pos'][i]
            head = p1['head']['pos'][i]
            hand_l = p1['left hand']['pos'][i]
            hand_r = p1['right hand']['pos'][i]
            foot_l = p1['left foot']['pos'][i]
            foot_r = p1['right foot']['pos'][i]
            
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(foot_l), (255, 0, 255), thickness=3, lineType=cv2.LINE_4)
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(foot_r), (255, 0, 255), thickness=3, lineType=cv2.LINE_4)
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(hand_l), (0, 255, 0), thickness=3, lineType=cv2.LINE_4)
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(hand_r), (0, 255, 0), thickness=3, lineType=cv2.LINE_4)

        #P2
        x1,y1 = p2['waist']['pos'][i-1][0],p2['waist']['pos'][i-1][1]
        x2,y2 = p2['waist']['pos'][i][0],p2['waist']['pos'][i][1]
        #yaw = quat2euler(p2.waist.pose[i])[2]

        if x1<0 or y1<0 or x2<0 or y2<0 or x1>size_px or y1>size_px or x2>size_px or y2>size_px:
            continue
                
        cv2.line(scene_map_tmp1, (x1, y1), (x2, y2), (255, 0, 0), thickness=3, lineType=cv2.LINE_4)
            
        if np.mod(i,10)==0 or i==p1['waist']['pos'].shape[0]-1:
                
            waist = p2['waist']['pos'][i]
            head = p2['head']['pos'][i]
            hand_l = p2['left hand']['pos'][i]
            hand_r = p2['right hand']['pos'][i]
            foot_l = p2['left foot']['pos'][i]
            foot_r = p2['right foot']['pos'][i]
                
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(foot_l), (255, 0, 255), thickness=3, lineType=cv2.LINE_4)
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(foot_r), (255, 0, 255), thickness=3, lineType=cv2.LINE_4)
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(hand_l), (0, 255, 0), thickness=3, lineType=cv2.LINE_4)
            cv2.line(scene_map_tmp2, tuple(((waist+head)/2).astype(np.int16)), tuple(hand_r), (0, 255, 0), thickness=3, lineType=cv2.LINE_4)

            cv2.imwrite('./viz_traj/scene_{:0=3}'.format(scene_id) + '/{:0=3}'.format(traj_id) + '_{:0=3}'.format(i) + '.png', scene_map_tmp2)
                        
        scene_map_tmp2 = copy.deepcopy(scene_map_tmp1)

def get_pospose(traj):
    
    traj_out = copy.deepcopy(traj)
    body = {'head','waist','right hand','left hand','right foot','left foot'}
    
    for b in body:
        traj_out[b]['pos'] = transform_axis_pos(traj[b]['pos'])
        traj_out[b]['pose'] = transform_axis_pose(traj[b]['pose'])
    
    return traj_out

def conv_scale(traj, img_center):
    
    traj_out = copy.deepcopy(traj)
    body = {'head','waist','right hand','left hand','right foot','left foot'}
    
    for b in body:
        traj_out[b]['pos'] = conv_meter2pixel(traj[b]['pos'], img_center)
    
    return traj_out
    
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
    img_center = get_center(path_boundary, scene_id)
    
    for i in range(len(data)):
        
        #extract trajectory data. traj1 and traj2 are trajectories for p1 and p2, and g1 is goal for p1.
        traj1 = data[i]['p1']
        traj2 = data[i]['p2']
        goal1 = data[i]['goal']
        
        #convert world coordinate[m] to image coordinate[pixel]
        p1 = conv_scale(traj1, img_center)
        p2 = conv_scale(traj2, img_center)
        g1 = conv_meter2pixel(goal1[0,:2], img_center)
        
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
    path_boundary = params['path_boundary']
    
    size_m = params['size_m']
    size_px = params['size_px']
    
    scene_id_list = params['scene_id']
    
    type_viz = params['type_viz']

    for scene_id in scene_id_list:
        print('visualizing scene ' + str(scene_id) + '...')
        disp_trajectory()