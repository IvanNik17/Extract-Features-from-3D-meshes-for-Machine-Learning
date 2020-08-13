# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:32:14 2020

Python Implementation of the suggested features in the paper "CLASSIFICATION OF AIRBORNE LASER SCANNING DATA USING GEOMETRIC MULTI-SCALE FEATURES AND DIFFERENT NEIGHBOURHOOD TYPES"

Code for extracting of Geometric Eigenvalue features, point cloud distribution features and statistical features used for training Random Forest Classifier for segmenting point clouds
@author: ivan nikolov
"""

import numpy as np

from open3d import *


import random

from sys import exit


    
def determinant_3x3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]) +
            m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]))


def subtract(a, b):
    return (a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2])

def tetrahedron_calc_volume(a, b, c, d):
    return (abs(determinant_3x3((subtract(a, b),
                                 subtract(b, c),
                                 subtract(c, d),
                                 ))) / 6.0)

meshPath = r"input"    
resultsPath = r"results"

# Load the point cloud and normals
if 'objVerts' not in  locals():
    bladeVerts = np.loadtxt(meshPath + r"\obj_vertices.txt", delimiter = ' ')

    
    
if 'objNormals' not in  locals():
    objNormals = np.loadtxt(meshPath + r"\obj_normals.txt", delimiter = ' ')



blade_verts = bladeVerts
blade_norms = objNormals

#  Transform the point cloud to the a PointCloud() object used by open3d
pcd = geometry.PointCloud()
    
pcd.points = utility.Vector3dVector(blade_verts)
pcd.normals = utility.Vector3dVector(blade_norms)

print(pcd)
# Calculate the KDTree
pcd_tree = geometry.KDTreeFlann(pcd)

allVerts = np.ones([len(blade_verts),1])

# how many scales will be used 
numScales = 1

# initialize a matrix that will contain all the vertices and the new calculated features
#  12 - shape measurements and 50 distribution features per area scale
blade_withFeatures = np.ones([len(blade_verts),4 + numScales*12 + numScales*50])

#  last column is left free so it can be used as a ground truth column
blade_withFeatures[:,4 + numScales*12 - 1 + numScales*50] *= 1



i=0

#  go through all the points
while i<len(blade_verts):
        
        pointScaleFeatures = []
        for j in [1]: #[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]
        
            #  for each radius area scale find neighbours
            [k_small, idx_small, distances_small] = pcd_tree.search_radius_vector_3d(pcd.points[i], j)
          
            currNdx = np.array(idx_small)
            #  if there are less than 2 neightbours just add 0s 
            if (len(currNdx) <=2):
                linearity= 0
                planarity=0
                sphericity=0
                omnivariance=0
                anisotropy=0
                eigenentropy=0
                sumOFEigs=0
                changeOfCurvature=0
                farthestDist =  distances_small[len(distances_small)-1]/j
                pointDensity = k_small/j
                heightStd=0
                heightMax=0
                
                shapeDist_curr = np.zeros(50)
            #  if there are enouigh neighbours then continue with computation
            else:
                
                #  get heighbourhood points and normals
                nearestNeighbors_normals = blade_norms[currNdx]
                nearestNeighbors_verts = blade_verts[currNdx]
                
                #  calculate the covariance matrix, and eigenvalues
                cov_mat = np.cov([nearestNeighbors_verts[:,0],nearestNeighbors_verts[:,1],nearestNeighbors_verts[:,2]])
                eig_val_cov, eig_vec_cov = np.linalg.eigh(cov_mat)
                idx = eig_val_cov.argsort()[::-1]  
                eig_val_cov = eig_val_cov[idx]
                
                #  calculate the first 12 features derived from shape measurements
                linearity = (eig_val_cov[0] - eig_val_cov[1])/eig_val_cov[0]
                planarity = (eig_val_cov[1] - eig_val_cov[2])/eig_val_cov[0]
                sphericity = eig_val_cov[2]/eig_val_cov[0]
                omnivariance =(eig_val_cov[0]*eig_val_cov[1]*eig_val_cov[2]) **(1./3.)
                anisotropy = (eig_val_cov[0] - eig_val_cov[2])/eig_val_cov[0]
                eigenentropy = -(( eig_val_cov[0] * np.log(eig_val_cov[0])) + ( eig_val_cov[1] * np.log(eig_val_cov[1])) + ( eig_val_cov[2] * np.log(eig_val_cov[2])))
                sumOFEigs = eig_val_cov[0]+ eig_val_cov[1]+eig_val_cov[2]
                changeOfCurvature = eig_val_cov[2]/(eig_val_cov[0]+ eig_val_cov[1]+eig_val_cov[2])
                
                farthestDist =  distances_small[len(distances_small)-1]/j
                pointDensity = k_small/j
                heightMax = np.abs(np.dot(nearestNeighbors_normals.mean(axis=0),nearestNeighbors_verts.T)).max() - np.abs(np.dot(nearestNeighbors_normals.mean(axis=0),nearestNeighbors_verts.T)).min()
                heightStd = np.abs(np.dot(nearestNeighbors_normals.mean(axis=0),nearestNeighbors_verts.T)).std()
                
                # calculate the distribution features - they are calculated for 255 random picks of points 
                centroid_vert = nearestNeighbors_verts.mean(axis=0)
                D1 = []
                D2 = []
                D3 = []
                D4 = []
                A3 = []
                for k in range(0,255):
                    
                    randomIndex = np.random.randint(0,len(nearestNeighbors_verts))
                    randPoint = nearestNeighbors_verts[randomIndex,:]
                    D1_curr = np.linalg.norm(centroid_vert-randPoint)
                    
                    
                    indices = random.sample(range(0,len(nearestNeighbors_verts)), 2)                   
                    randPoint1 = nearestNeighbors_verts[indices[0],:]
                    randPoint2 = nearestNeighbors_verts[indices[1],:]
                    D2_curr = np.linalg.norm(randPoint1-randPoint2)
                    
                    
                    indices = random.sample(range(0,len(nearestNeighbors_verts)), 3)  
                    randPoint1 = nearestNeighbors_verts[indices[0],:]
                    randPoint2 = nearestNeighbors_verts[indices[1],:]
                    randPoint3 = nearestNeighbors_verts[indices[2],:]
                    vec1 = randPoint1 - randPoint2
                    vec2 = randPoint1 - randPoint3
                    area = np.linalg.norm(np.cross(vec1,vec2))/2
                    D3_curr = np.sqrt(area)
                    
                    
                    indices = random.sample(range(0,len(nearestNeighbors_verts)), 3) 
                    randPoint1 = nearestNeighbors_verts[indices[0],:]
                    randPoint2 = nearestNeighbors_verts[indices[1],:]
                    randPoint3 = nearestNeighbors_verts[indices[2],:]
                    vec1 = randPoint1 - randPoint2
                    vec2 = randPoint1 - randPoint3
                    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    A3_curr = np.arccos(cosine_angle)
                    
                    
                    indices = random.sample(range(0,len(nearestNeighbors_verts)), 4) 
                    randPoint1 = nearestNeighbors_verts[indices[0],:]
                    randPoint2 = nearestNeighbors_verts[indices[1],:]
                    randPoint3 = nearestNeighbors_verts[indices[2],:]
                    randPoint4 = nearestNeighbors_verts[indices[3],:]
                    volTetra = tetrahedron_calc_volume(randPoint1, randPoint2, randPoint3, randPoint4)
                    D4_curr = np.cbrt(volTetra)
                    
                    D1.append(D1_curr)
                    D2.append(D2_curr)
                    D3.append(D3_curr)
                    D4.append(D4_curr)
                    A3.append(A3_curr)
                # The 255 results for each features are used to calculate the histogram distribution of it in 10 bins 
                counts_D1, bin_edges = np.histogram(np.array(D1), bins=10)
                counts_D2, bin_edges = np.histogram(np.array(D2), bins=10)
                counts_D3, bin_edges = np.histogram(np.array(D3), bins=10)
                counts_D4, bin_edges = np.histogram(np.array(D4), bins=10)
                counts_A3, bin_edges = np.histogram(np.array(A3), bins=10)
                
                
                #  These distributions are then saved
                shapeDist_curr = np.concatenate((counts_D1,counts_D2,counts_D3,counts_D4,counts_A3), axis=0)  
            #  Added to the other features and a feature vector is created
            pointScaleFeatures.extend([linearity,planarity,sphericity,omnivariance,anisotropy,eigenentropy,sumOFEigs,changeOfCurvature,farthestDist,pointDensity,heightMax,heightStd])
            pointScaleFeatures.extend(shapeDist_curr.tolist())
        #  The feature vector for each point is checked for NaN values and then concatenated
        allVals = [blade_verts[i,0],blade_verts[i,1],blade_verts[i,2]]
        where_are_NaNs = np.isnan(pointScaleFeatures)
        pointScaleFeatures = np.array(pointScaleFeatures)
        pointScaleFeatures[where_are_NaNs] = 0
        pointScaleFeatures = pointScaleFeatures.tolist()
        allVals.extend(pointScaleFeatures)
        blade_withFeatures[i,:-1] = allVals

        
        
        
        i+=1
        
np.savetxt(resultsPath + r"\\"+ "obj"  + ".txt", blade_withFeatures,delimiter=',')   