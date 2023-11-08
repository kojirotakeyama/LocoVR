import numpy as np
import os
import cv2
import open3d as o3d
import copy
from pygltflib import GLTF2
from pygltflib.utils import ImageFormat
import yaml

unknown_col = [0,0,0]

#thresholds for traversable map generation 1
thre_trvs_min = -0.2
thre_trvs_max = 0.15

thre_h = [-0.2,0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.0]

def get_corner(path, scene_id):
    
    data = np.genfromtxt(path ,delimiter=',')
    
    c1 = np.zeros(3)
    c2 = np.zeros(3)
    
    if data[:,0].tolist().count(scene_id)>0:
        idx = data[:,0].tolist().index(scene_id)
        c1 = data[idx,1:4]
        c2 = data[idx,4:7]
    
    return c1, c2

def get_grid_img(p):
    
    p_grid = []
    
    x_min = (np.floor(np.min(p[:,0]))).astype(np.int32)
    x_max = (np.ceil(np.max(p[:,0]))).astype(np.int32)
    y_min = (np.floor(np.min(p[:,1]))).astype(np.int32)
    y_max = (np.ceil(np.max(p[:,1]))).astype(np.int32)
    
    for x in range(x_min,x_max):
        for y in range(y_min,y_max):
            
            area = 0.5*(-p[1,1]*p[2,0] + p[0,1]*(-p[1,0] + p[2,0]) + p[0,0]*(p[1,1] - p[2,1]) + p[1,0]*p[2,1])
            s = 1/(2*area)*(p[0,1]*p[2,0] - p[0,0]*p[2,1] + (p[2,1] - p[0,1])*x + (p[0,0] - p[2,0])*y)
            t = 1/(2*area)*(p[0,0]*p[1,1] - p[0,1]*p[1,0] + (p[0,1] - p[1,1])*x + (p[1,0] - p[0,0])*y)
            
            if ((0 < s < 1) and (0 < t < 1) and (0 < 1-s-t < 1)):
                p_grid.append(np.array([x,y]))
                
    return np.array(p_grid)

def update_img_h(img_h, p, p_h):
    
    img_h_out = copy.deepcopy(img_h)
    
    if p.shape[0]==0:
        
        return img_h_out
    
    for i in range(p.shape[0]):
        
        if p[i,0]<0 or p[i,0]>=1024 or p[i,1]<0 or p[i,1]>=1024:
            continue
            
        if img_h[p[i,0],p[i,1]] > p_h[i]:
            continue
        
        img_h_out[p[i,0],p[i,1]] = p_h[i]
    
    return img_h_out

def get_height_grid(p_g, p_m, p_m_h):
    
    p_g_h = []
    
    if p_g.shape[0] == 0:
        return np.array([])
    
    for i in range(p_g.shape[0]):
        v1 = p_m[1,:] - p_m[0,:]
        v2 = p_m[2,:] - p_m[0,:]
        v3 = p_g[i] - p_m[0,:]
        
        a = (v3[0]*v2[1] - v2[0]*v3[1])/(v1[0]*v2[1] - v1[1]*v2[0])
        b = (v1[1]*v3[0] - v1[0]*v3[1])/(v1[1]*v2[0] - v1[0]*v2[1])
        
        h1 = p_m_h[1] - p_m_h[0]
        h2 = p_m_h[2] - p_m_h[0]
        
        h3 = a*h1 + b*h2
        p_g_h.append(p_m_h[0] + h3)
    
    return np.array(p_g_h)

def make_traversable_map(img_h, floor_height):
    
    img_h2 = img_h*0+255
    img_h2[img_h-floor_height<thre_trvs_min] = 0
    img_h2[img_h-floor_height>thre_trvs_max] = 0
    
    return img_h2

def make_height_map(img_h, floor_height):
    
    img_h2 = img_h*0+200
    img_h2[img_h-floor_height<thre_h[0]] = 255
    
    for i in range(len(thre_h)-1):
        
        img_h2[img_h-floor_height>thre_h[i]] = 200-np.floor(200/(len(thre_h))-1)*i
    
    img_h2[img_h-floor_height>thre_h[-1]] = 0
    
    return img_h2

def generate_height_map(load_dir, fname):
    
    scene_id = int(fname[2:5])
    
    path_mesh = load_dir + fname + '/' + fname[6:17] + '.semantic.glb'

    path_save = './height_map/'    
    corner1, corner2 = get_corner("./boundary.txt", scene_id)
    
    if np.sum(abs(corner1)) == 0 or np.sum(abs(corner2)) == 0:
        return 0
    
    xlim = [corner1[0], corner2[0]]
    ylim = [corner1[1], corner2[1]]
    zlim = [corner1[2]-1,corner2[2]+1.5]

    mesh = o3d.io.read_triangle_mesh(path_mesh)
    mesh.compute_vertex_normals()
    
    triangle = np.asarray(mesh.triangles)
    id_cut = []
    
    for i in range(triangle.shape[0]):
        
        if np.min(np.asarray(mesh.vertices)[triangle[i]][:,0]) > xlim[1] + 1 or np.max(np.asarray(mesh.vertices)[triangle[i]][:,0]) < xlim[0] - 1:
            continue
        if np.min(np.asarray(mesh.vertices)[triangle[i]][:,1]) > ylim[1] + 1 or np.max(np.asarray(mesh.vertices)[triangle[i]][:,1]) < ylim[0] - 1:
            continue
        if np.min(np.asarray(mesh.vertices)[triangle[i]][:,2]) > zlim[1]  or np.max(np.asarray(mesh.vertices)[triangle[i]][:,2]) < zlim[0]:
            continue
        
        id_cut.append(i)
    
    vertices = np.asarray(mesh.vertices)
    
    p_mesh_o = np.zeros([3,2])
    img_h = np.zeros([size_px[0],size_px[1]])-100

    room_center=np.zeros(3)
    room_center[0] = (xlim[0]+xlim[1])/2
    room_center[1] = (ylim[0]+ylim[1])/2
    room_center[2] = corner1[2]
    
    for i in range(len(id_cut)):
        print('processing... ' + str(i) + ' / ' + str(len(id_cut)))
        p = vertices[triangle[id_cut[i]]]
        p_mesh_o[0,:] = (p[0,0:2]-room_center[0:2]+size_m/2)*size_px/size_m
        p_mesh_o[1,:] = (p[1,0:2]-room_center[0:2]+size_m/2)*size_px/size_m
        p_mesh_o[2,:] = (p[2,0:2]-room_center[0:2]+size_m/2)*size_px/size_m
        
        p_grid_o = get_grid_img(p_mesh_o)
        
        h_grid_o = get_height_grid(p_grid_o, p_mesh_o, p[:,2])
        img_h = update_img_h(img_h, p_grid_o, h_grid_o)

    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    img_h2 = make_height_map(img_h, room_center[2])
    cv2.imwrite(path_save + fname[2:5] + '.png', np.swapaxes(img_h2,0,1))

def generate_binary_map(load_dir, fname):
    
    scene_id = int(fname[2:5])
    
    #path_mesh = load_dir + '3D/' + fname + '/' + fname[6:17] + '.obj'
    path_mesh = load_dir + fname + '/' + fname[6:17] + '.semantic.glb'
    
    path_save = './binary_map/'    
    
    #Get boundary area of the room (corner1: upper left, corner2: bottom right)
    corner1, corner2 = get_corner("./boundary.txt", scene_id)
    
    if np.sum(abs(corner1)) == 0 or np.sum(abs(corner2)) == 0:
        return 0
    
    xlim = [corner1[0], corner2[0]]
    ylim = [corner1[1], corner2[1]]
    zlim = [corner1[2]-1,corner2[2]+1.5]    #third element indicates floor height of the room

    mesh = o3d.io.read_triangle_mesh(path_mesh)
    mesh.compute_vertex_normals()
    
    #Get mesh triangles
    triangle = np.asarray(mesh.triangles)
    id_cut = []
    
    #Extract mesh ids in the specified boundary area
    for i in range(triangle.shape[0]):
        
        if np.min(np.asarray(mesh.vertices)[triangle[i]][:,0]) > xlim[1] + 1 or np.max(np.asarray(mesh.vertices)[triangle[i]][:,0]) < xlim[0] - 1:
            continue
        if np.min(np.asarray(mesh.vertices)[triangle[i]][:,1]) > ylim[1] + 1 or np.max(np.asarray(mesh.vertices)[triangle[i]][:,1]) < ylim[0] - 1:
            continue
        if np.min(np.asarray(mesh.vertices)[triangle[i]][:,2]) > zlim[1]  or np.max(np.asarray(mesh.vertices)[triangle[i]][:,2]) < zlim[0]:
            continue
        
        id_cut.append(i)
    
    #Get vertices of mesh triangles
    vertices = np.asarray(mesh.vertices)
    
    p_mesh_o = np.zeros([3,2])
    img_h = np.zeros([size_px[0],size_px[1]])-100

    #Calculate center point from boundaries
    room_center=np.zeros(3)
    room_center[0] = (xlim[0]+xlim[1])/2
    room_center[1] = (ylim[0]+ylim[1])/2
    room_center[2] = corner1[2]
    
    for i in range(len(id_cut)):
        print('processing... ' + str(i) + ' / ' + str(len(id_cut)))
        
        #Mesh vertices in World coordinate
        p = vertices[triangle[id_cut[i]]]
        
        #Mesh vertices in image coordinate
        p_mesh_o[0,:] = (p[0,0:2]-room_center[0:2]+size_m/2)*size_px/size_m
        p_mesh_o[1,:] = (p[1,0:2]-room_center[0:2]+size_m/2)*size_px/size_m
        p_mesh_o[2,:] = (p[2,0:2]-room_center[0:2]+size_m/2)*size_px/size_m
        
        #Points in the mesh triangle in image coordinate
        p_grid_o = get_grid_img(p_mesh_o)
        
        #Height image
        h_grid_o = get_height_grid(p_grid_o, p_mesh_o, p[:,2])
        
        #Get height of each image pixel from mesh triangle
        img_h = update_img_h(img_h, p_grid_o, h_grid_o)

    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    #Generate traversable map by thresholding the height
    img_h2 = make_traversable_map(img_h, room_center[2])
    
    #Save image
    cv2.imwrite(path_save + fname[2:5] + '.png', np.swapaxes(img_h2,0,1))

if __name__ == "__main__":

    with open('./config.yaml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        
    load_dir_base = params['path_habitat']
    
    size_px = np.array([params['size_px'],params['size_px']])
    size_m = np.array([params['size_m'],params['size_m']])
    
    fname = os.listdir(load_dir_base)
    
    for i in range(len(fname)):
        
        if params['flag_binary']==1:
            generate_binary_map(load_dir_base + '/', fname[i])
        if params['flag_height']==1:
            generate_height_map(load_dir_base + '/', fname[i])
    
        
        
    
    