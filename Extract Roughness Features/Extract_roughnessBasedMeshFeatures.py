# -*- coding: utf-8 -*-
"""
Created on Thu June  6 15:09:54 2020

Code for extracting mesh-based features used for classifying mesh surfaces based on their roughness properties. 

@author: ivan nikolov
"""

import numpy as np


from roughnessFunctions import *

from open3d import *

import vispy.scene
from vispy.scene import visuals


def visualize3D(titleV,pointCloud, rgbMap):
    
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title = titleV)
    view = canvas.central_widget.add_view()
    
    
    scatter = visuals.Markers()
    scatter.set_data(pointCloud, edge_width=0.1, edge_color=rgbMap, face_color=rgbMap, size=5, scaling=False)
    
    view.add(scatter)
    
    view.camera = 'arcball'
     

meshPath = r"input"    

# The mesh is separated into four files containing vertices, faces, normals and colors, but the same can be done 
# by using .ply files and using open3d's object import functions

if 'objVerts' not in  locals():
    bladeVerts = np.loadtxt(meshPath + r"\object_vertices.txt", delimiter = ' ')
    
    
if 'objFaces' not in  locals():
    bladeFaces = np.loadtxt(meshPath + r"\object_faces.txt", delimiter = ' ')
    
    
if 'objNormals' not in  locals():
    objNormals = np.loadtxt(meshPath + r"\object_normals.txt", delimiter = ' ')
    
if 'objColor' not in  locals():
    objColor = np.loadtxt(meshPath + r"\object_colors.txt", delimiter = ' ')




blade_verts = bladeVerts
blade_colors = objColor
blade_norms = objNormals
blade_faces = bladeFaces
blade_faces = blade_faces.astype(int)


#Parameters - hyper parameters for extracting the features - mostly changing the area around each vertex which is used to calculate them

#Saliency:
saliency_a= 1;

#Entropy:
entropy_knn = 80

#Difference of Normals:
don_radPercent = 0.02


#Point Density:
pdcu_percentChange = [0.1,0.15,0.15]
pdcu_num_percentChange = 0.5
pdcu_percentFromMaxNeighbour = 0.3


# Calculate mesh features - some times the calculations may contain Nan values, especially if the cloud/mesh is too sparse or it has noise

saliencyMap = meshSaliency(blade_verts, blade_faces,saliency_a)
saliencyNonNan = np.isnan(saliencyMap)
saliencyMap[saliencyNonNan[:,0],:] = 0


entropyMap = meshEntropy(blade_verts, blade_colors,entropy_knn)
entropyNonNan = np.isnan(entropyMap)
entropyMap[entropyNonNan[:,0],:] = 0


donMap = meshDifferenceOfNormals(blade_verts, blade_norms,don_radPercent)
donMapNonNan = np.isnan(donMap)
donMap[donMapNonNan[:,0],:] = 0


pdcuMap = meshPointDensityAndColorUniformity(blade_verts,pdcu_percentFromMaxNeighbour)
pdcuMapNonNan = np.isnan(pdcuMap)
pdcuMap[pdcuMapNonNan[:,0],:] = 0

# Output containing the vertices in a X,Y,Z column together with the four calculated features
outputFeatures = np.concatenate([blade_verts, saliencyMap, donMap, entropyMap, pdcuMap], axis=1)




# Visualize using Vispy - comment out if you do not have it installed or not want to visualize point clouds/meshes
visualize3D("Colored Model",blade_verts, blade_colors/255)


colorMap = np.ones([len(entropyMap_norm),1])
image_rgb = np.concatenate([entropyMap_norm,entropyMap_norm,entropyMap_norm,colorMap], axis=1)
visualize3D("Color Entropy", blade_verts, image_rgb)

colorMap = np.ones([len(saliencyMap_norm),1])
image_rgb = np.concatenate([saliencyMap_norm,saliencyMap_norm,saliencyMap_norm,colorMap], axis=1)
visualize3D("Color Entropy", blade_verts, image_rgb)

colorMap = np.ones([len(donMap_norm),1])
image_rgb = np.concatenate([donMap_norm,donMap_norm,donMap_norm,colorMap], axis=1)
visualize3D("Color Entropy", blade_verts, image_rgb)

colorMap = np.ones([len(pdcuMap_norm),1])
image_rgb = np.concatenate([pdcuMap_norm,pdcuMap_norm,pdcuMap_norm,colorMap], axis=1)
visualize3D("Color Entropy", blade_verts, image_rgb)
