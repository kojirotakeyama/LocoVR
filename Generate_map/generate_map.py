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
    
def conv_bgr2hex(bgr):
    
    return '%02X%02X%02X' % (bgr[2],bgr[1],bgr[0])

def get_trans_type(path, scene_id):
    
    #"data" contains scene_id in the first row and transformation type in the second row.
    data = np.genfromtxt(path ,delimiter=',')
    
    idx = np.where(data[:,0]==scene_id)
    
    if idx[0].shape[0]==0:
        
        trans_type = 0
    else:
        trans_type = int(data[idx][0,1])
    
    return trans_type


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

def conv_o2tx(p_grid, p_mesh, uv):
    
    p_grid_tx = []
    
    if p_grid.shape[0] == 0:
        return np.array([])
    
    for i in range(p_grid.shape[0]):
        v1 = p_mesh[1,:] - p_mesh[0,:]
        v2 = p_mesh[2,:] - p_mesh[0,:]
        v3 = p_grid[i] - p_mesh[0,:]
        
        a = (v3[0]*v2[1] - v2[0]*v3[1])/(v1[0]*v2[1] - v1[1]*v2[0])
        b = (v1[1]*v3[0] - v1[0]*v3[1])/(v1[1]*v2[0] - v1[0]*v2[1])
        
        V1 = uv[1,:] - uv[0,:]
        V2 = uv[2,:] - uv[0,:]
        
        V3 = np.array([a*V1[0] + b*V2[0], a*V1[1] + b*V2[1]])
        p_grid_tx.append(uv[0,:] + V3)
    
    return np.array(p_grid_tx).astype(np.int32)

def get_color(p, img):
    
    colors = []
    
    for i in range(p.shape[0]):
        
        if p[i,0]<0 or p[i,0]>=2048 or p[i,1]<0 or p[i,1]>=2048:
            
            colors.append(np.array([-1,-1,-1]))
        
        else:
            colors.append(img[p[i,0],p[i,1],:])
    
    return np.array(colors)

def update_img_ch(img_o, img_h, p, colors, p_h):
    
    img_o_out = copy.deepcopy(img_o)
    img_h_out = copy.deepcopy(img_h)
    
    if p.shape[0]==0 or colors.shape[0]==0:
        
        return img_o_out, img_h_out
    
    for i in range(p.shape[0]):
        
        if np.min(colors[i]) == -1:
            continue
        
        if p[i,0]<0 or p[i,0]>=1024 or p[i,1]<0 or p[i,1]>=1024:
            continue
            
        if img_h[p[i,0],p[i,1]] > p_h[i]:
            continue
        
        img_o_out[p[i,0],p[i,1],:] = colors[i]
        img_h_out[p[i,0],p[i,1]] = p_h[i]
    
    return img_o_out, img_h_out
        
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

def load_label(label_name):
    
    tmp = np.genfromtxt(label_name, delimiter='    ', skip_header=1, dtype=None, encoding='utf-8')
    label_list1 = []
    label_list2 = []
    label_list3 = []
    label_list4 = []
    
    for i in range(len(tmp)):
        label_list1.append(str(list(tmp[i])[1]).replace('"',''))
        label_list2.append(str(list(tmp[i])[2]).replace('"',''))
        label_list3.append(list(tmp[i])[4])
        label_list4.append(list(tmp[i])[5])
    
    return [label_list1,label_list2,label_list3,label_list4]

def conv_id2sem(id2sem):
    
    color_code = []
    semantics = []
    
    for i in range(len(id2sem)):
        
        color_code.append(str(id2sem[i][1]))
        semantics.append(str(id2sem[i][2]).replace('"',''))
    
    return [color_code,semantics]

def generate_list_sem2semcol(sem, color_type):
    
    col = []
    
    for i in range(len(sem[1])):
        
        if color_type == 0: #color by original semantic index
            seed = i
        
        if color_type == 1: #color by nyuid
            seed = sem[2][i]
        
        if color_type == 2: #color by nyuid40
            seed = sem[3][i]
        
        rgb = np.array([np.mod(seed*311,255),np.mod(seed*613+100,255),np.mod(seed*97+155,255)])
            
        col.append(rgb)
    
    return list([sem, col])
    
def generate_list_id2semcol(id2sem, sem2semcol):
    
    semcol = []
    
    for i in range(len(id2sem[0])):
        
        key = id2sem[1][i]
        col = np.array([0,0,0])
        
        if key in sem2semcol[0][0]:
            idx = sem2semcol[0][0].index(key)
            col = sem2semcol[1][idx]
        
        elif key in sem2semcol[0][1]:
            idx = sem2semcol[0][1].index(key)
            col = sem2semcol[1][idx]
            
        semcol.append(col)

    return [id2sem[0], semcol]

def get_id2semcol(id2sem, sem_list, color_type):
    
    id2sem = conv_id2sem(id2sem)
    sem2semcol = generate_list_sem2semcol(sem_list, color_type)
    id2semcol = generate_list_id2semcol(id2sem, sem2semcol)
    
    return id2semcol
    
def conv_id2semcol(color, id2semcol):
    
    semcol = []
    
    for i in range(color.shape[0]):
        
        col_hex = conv_bgr2hex(color[i])
        
        if col_hex in id2semcol[0]:
            idx = id2semcol[0].index(col_hex)
            semcol.append(id2semcol[1][idx])
        else:
            semcol.append(np.array(unknown_col))
    
    return np.array(semcol)
    
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

def make_texturefiles_from_mesh(path_mesh, path_texture):
    
    if not os.path.exists(path_texture):
        os.makedirs(path_texture)
        
        gltf = GLTF2().load(path_mesh)
        gltf.images[0].uri
        gltf.convert_images(ImageFormat.FILE, path = path_texture)
    
        fname = os.listdir(path_texture)
        
        for i in range(len(fname)):
            if len(fname[i])==5:
                os.rename(path_texture + fname[i], path_texture + '00' + fname[i])
            if len(fname[i])==6:
                os.rename(path_texture + fname[i], path_texture + '0' + fname[i]) 
    
def get_texture_from_file(path):
    
    fname = sorted(os.listdir(path))
    img = []
    
    for i in range(len(fname)):
        img.append(np.swapaxes(cv2.imread(path+fname[i]),0,1)[:,::-1,:])
    
    return img

def get_texture_from_file2(path):
    
    fname = sorted(os.listdir(path))
    img = []
    
    for i in range(len(fname)):
        
        if not ".jpg" in fname[i]:
            continue
        img.append(np.swapaxes(cv2.imread(path+fname[i]),0,1)[:,::-1,:])
    
    return img

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
    cv2.imwrite(path_save + fname[2:5] + '.png', img_h2)

def generate_texture_map(load_dir, fname):
    
    scene_id = int(fname[2:5])
    
    path_mesh = load_dir + fname + '/' + fname[6:17] + '.obj'
    path_texture2 = load_dir + fname + '/'
    path_save = './texture_map/'
        
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
    
    uv = np.asarray(mesh.triangle_uvs)
    m_ids = np.asarray(mesh.triangle_material_ids)-1
    vertices = np.asarray(mesh.vertices)
    
    p_mesh_o = np.zeros([3,2])
    img_o = np.zeros([size_px[0],size_px[1],3])
    img_h = np.zeros([size_px[0],size_px[1]])-100

    img_tx = get_texture_from_file2(path_texture2)
    
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
        p_grid_tx = conv_o2tx(p_grid_o, p_mesh_o, uv[id_cut[i]*3:id_cut[i]*3+3]*2048)
        color_grid_o = get_color(p_grid_tx, img_tx[m_ids[id_cut[i]]])

        h_grid_o = get_height_grid(p_grid_o, p_mesh_o, p[:,2])
        img_o, img_h = update_img_ch(img_o, img_h, p_grid_o, color_grid_o, h_grid_o)

    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    cv2.imwrite(path_save + fname[2:5] + '.png', img_o)

def generate_semantic_map(load_dir, fname):
    
    color_type = 2  #coloring strategy for semantic label, 0: color by original semantic index 1: #color by nyuid 2: #color by nyu40id
    
    scene_id = int(fname[2:5])
    
    path_mesh = load_dir + fname + '/' + fname[6:17] + '.semantic.glb'
    
    path_texture = './save_semantic_textures/{:0=3}/'.format(scene_id)
    path_save = './semantic_map/'
        
    corner1, corner2 = get_corner("./boundary.txt", scene_id)
    
    id2sem = np.genfromtxt(load_dir + fname + '/' + fname[6:17] + ".semantic.txt", delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
    sem_list = load_label('./semantic_label.txt')
    
    id2semcol = get_id2semcol(id2sem, sem_list, color_type)
    
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
    
    uv = np.asarray(mesh.triangle_uvs)
    m_ids = np.asarray(mesh.triangle_material_ids)-1
    vertices = np.asarray(mesh.vertices)
    
    p_mesh_o = np.zeros([3,2])
    img_o = np.zeros([size_px[0],size_px[1],3])
    img_h = np.zeros([size_px[0],size_px[1]])-100

    make_texturefiles_from_mesh(path_mesh, path_texture)
    img_tx = get_texture_from_file(path_texture)
    
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
        p_grid_tx = conv_o2tx(p_grid_o, p_mesh_o, uv[id_cut[i]*3:id_cut[i]*3+3]*2048)
        color_grid_o = get_color(p_grid_tx, img_tx[m_ids[id_cut[i]]])
        
        color_grid_o = conv_id2semcol(color_grid_o, id2semcol)
        
        h_grid_o = get_height_grid(p_grid_o, p_mesh_o, p[:,2])
        img_o, img_h = update_img_ch(img_o, img_h, p_grid_o, color_grid_o, h_grid_o)

    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    cv2.imwrite(path_save + fname[2:5] + '.png', img_o)

def generate_binary_map(load_dir, fname):
    
    scene_id = int(fname[2:5])
    
    #path_mesh = load_dir + '3D/' + fname + '/' + fname[6:17] + '.obj'
    path_mesh = load_dir + fname + '/' + fname[6:17] + '.semantic.glb'
    
    path_save = './binary_map/'    
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
    
    img_h2 = make_traversable_map(img_h, room_center[2])
    cv2.imwrite(path_save + fname[2:5] + '.png', img_h2)

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
        if params['flag_semantic']==1:
            generate_semantic_map(load_dir_base + '/', fname[i])
        if params['flag_texture']==1:
            generate_texture_map(load_dir_base + '/', fname[i])
    
        
        
    
    