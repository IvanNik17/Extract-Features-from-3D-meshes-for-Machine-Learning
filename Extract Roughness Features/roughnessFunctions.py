# -*- coding: utf-8 -*-
"""
Created on Thu June  6 14:17:09 2020

@author: ivan nikolov
"""

import numpy as np
import math
from scipy.sparse import csr_matrix, spdiags, find
from open3d import *



def crossinline(x,y):
    z = x.copy()
    z[:,0] = x[:,1]*y[:,2] - x[:,2]*y[:,1]
    z[:,1] = x[:,2]*y[:,0] - x[:,0]*y[:,2]
    z[:,2] = x[:,0]*y[:,1] - x[:,1]*y[:,0]

    return z
    
    
def get_entropy(signal):
    """ Uses log2 as base
    """
    probabability_distribution = [np.size(signal[signal == i])/(1.0 * signal.size) for i in list(set(signal))]
    entropy = np.sum([pp * np.log2(1.0 / pp) for pp in probabability_distribution])
    return entropy

def rgb_to_hsv(rgb):
    """
    >>> from colorsys import rgb_to_hsv as rgb_to_hsv_single
    >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(50, 120, 239))
    'h=0.60 s=0.79 v=239.00'
    >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(163, 200, 130))
    'h=0.25 s=0.35 v=200.00'
    >>> np.set_printoptions(2)
    >>> rgb_to_hsv(np.array([[[50, 120, 239], [163, 200, 130]]]))
    array([[[   0.6 ,    0.79,  239.  ],
            [   0.25,    0.35,  200.  ]]])
    >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(100, 100, 100))
    'h=0.00 s=0.00 v=100.00'
    >>> rgb_to_hsv(np.array([[50, 120, 239], [100, 100, 100]]))
    array([[   0.6 ,    0.79,  239.  ],
           [   0.  ,    0.  ,  100.  ]])
    """
    input_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc

    deltac = maxc - minc
    s = deltac / maxc
    deltac[deltac == 0] = 1  # to not divide by zero (those results in any way would be overridden in next lines)
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    h = 4.0 + gc - rc
    h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
    h[r == maxc] = bc[r == maxc] - gc[r == maxc]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0
    res = np.dstack([h, s, v])
    return res.reshape(input_shape)


    
    # Metric translated in Python from the Matlab implementation presented in http://www.gipsa-lab.grenoble-inp.fr/~kai.wang/papers/CG12.pdf
def meshSaliency(blade_verts, blade_faces, a = 0.15):


    blade_faces = blade_faces.astype(int)
    
    
    numVerts = len(blade_verts)
    numFaces = len(blade_faces)
    
    #to store the angles sum on vertices
    sum_angles = np.zeros([numVerts,1])
    
    #to store the facet areas
    area = np.zeros([numFaces,1])
    
    #for numerical computation stability
    epsilon = 1e-10
    
    blade_verts = blade_verts.T
    blade_faces = blade_faces.T
    
    for i in [1,2,3]:
        i1 = np.mod(i-1,3)
        i2 = np.mod(i,3)
        i3 = np.mod(i+1,3)
    
    #    pp = vertices(:,faces(i2,:)) - vertices(:,faces(i1,:));
        pp = blade_verts[:,blade_faces[i2,:]] - blade_verts[:, blade_faces[i1,:]]
    
    #   qq = vertices(:,faces(i3,:)) - vertices(:,faces(i1,:));
        qq = blade_verts[:,blade_faces[i3,:]] - blade_verts[:, blade_faces[i1,:]]
        
        
    #    normalize the vectors

        pp_length = np.sqrt(np.sum(pp**2,axis=0))
        qq_length = np.sqrt(np.sum(qq**2,axis=0))
        
    
        Ipp_zero = (pp_length<epsilon).nonzero()
        pp_length[Ipp_zero] = 1       
    
        Iqq_zero = (qq_length<epsilon).nonzero() 
        qq_length[Iqq_zero] = 1
        
    
    
    
        pp_nor = pp/ np.kron(np.ones((3,1)),pp_length)
        qq_nor = qq/ np.kron(np.ones((3,1)),qq_length)
    
    
    #    compute angles and clamped cotans
    
        cos_ang = np.sum(pp_nor*qq_nor,axis=0)
        cos_ang = np.clip(cos_ang,-1,1)
        ang = np.arccos(cos_ang)
        print(i)
        if i == 1:    
            ctan_1 = (1/np.tan(ang))/2
            ctan_1 = np.clip(ctan_1,0.001,1000)
            
            ii_lap_1 = blade_faces[i2,:]
            jj_lap_1 = blade_faces[i3,:]
        elif i == 2:
            ctan_2 = (1/np.tan(ang))/2
            ctan_2 = np.clip(ctan_2,0.001,1000)
            
            ii_lap_2 = blade_faces[i2,:]
            jj_lap_2 = blade_faces[i3,:]
        elif i == 3:
            ctan_3 = (1/np.tan(ang))/2
            ctan_3 = np.clip(ctan_3,0.001,1000)
            
            ii_lap_3 = blade_faces[i2,:]
            jj_lap_3 = blade_faces[i3,:]
    
    #accumulate the angles on vertices
        
        for j in range(0,numFaces):
            indextemp = blade_faces[i1,j]
            sum_angles[indextemp,0] = sum_angles[indextemp,0] + ang[j]
    
    #compute the facet areas
        if i == 1:
    
            rr = crossinline(pp.T,-qq.T)
            rr = rr.T
            area = np.sqrt(np.sum(rr**2,axis=0))/2
    
    # Laplacian matrix (stiffness matrix)
    ii_lap = np.concatenate([ii_lap_1, jj_lap_1, ii_lap_2, jj_lap_2, ii_lap_3, jj_lap_3])
    jj_lap = np.concatenate([jj_lap_1, ii_lap_1, jj_lap_2, ii_lap_2, jj_lap_3, ii_lap_3])
    ss_lap = np.concatenate([ctan_1, ctan_1, ctan_2, ctan_2, ctan_3, ctan_3])
    
    laplacian = csr_matrix( (ss_lap,(ii_lap, jj_lap))  , shape = (numVerts,numVerts), dtype=np.float)
    laplacian.eliminate_zeros()
    
    diag_laplacian = np.sum(laplacian,axis=0)
    
    Diag_laplacian = spdiags(diag_laplacian[:],0,numVerts,numVerts)
    
    laplacian = Diag_laplacian - laplacian
    
    #lumped mass matrix
    ii_mass = np.concatenate([blade_faces[0,:], blade_faces[1,:], blade_faces[2,:]])
    jj_mass = np.concatenate([blade_faces[0,:], blade_faces[1,:], blade_faces[2,:]])
    area = area / 3.0
    ss_mass =np.concatenate( [area,area,area] )
    
    mass = csr_matrix( (ss_mass,(ii_mass, jj_mass))  , shape = (numVerts,numVerts), dtype=np.float)
    mass.eliminate_zeros()
    
    #facet-edge adjacency matrix
    ii_adja = np.concatenate([blade_faces[0,:], blade_faces[1,:], blade_faces[2,:]])
    jj_adja = np.concatenate([blade_faces[1,:], blade_faces[2,:], blade_faces[0,:]])
    ss_adja = np.concatenate( [np.arange(0,numFaces), np.arange(0,numFaces), np.arange(0,numFaces)] )
    adja = csr_matrix( (ss_adja,(ii_adja, jj_adja))  , shape = (numVerts,numVerts), dtype=np.float)
    
    
    #add missing points
    #I_adja = find(csr_matrix.transpose(adja)!=0)
    arow,acol,adata = find(csr_matrix.transpose(adja)!=0)
    onlyLeft = adja[arow,acol] ==0
    onlyLeft = np.asarray(onlyLeft)
    onlyLeft = onlyLeft.T
    arow = arow[onlyLeft[:,-1]]
    acol = acol[onlyLeft[:,-1]]
    adja[arow,acol] = -1
    
    #find the boundary
    I,J,V = find(adja)
    I = I[V==-1]
    J = J[V==-1]
    
    flag_boundary = np.zeros([numVerts,1])
    flag_boundary[I,0] = 1
    flag_boundary[J,0] = 1
    
    #set different constant values (for Gaussian curvature computation) for
    #boundary vertices and non-boundary vertices
    constants = np.ones([numVerts,1])*2*math.pi
    I_boundary = np.where(flag_boundary == 1)[0]
    constants[I_boundary] = np.ones([len(I_boundary),1])*math.pi
    
    cgauss = constants - sum_angles
    
    
    surface = np.sum(mass.diagonal())
    
    L = laplacian
    
    L_diag = L.diagonal(0)
    
    cgaussabs = np.abs(cgauss)
    
    cgauss_rough = cgaussabs.T * L
    
    cgauss_rough = cgauss_rough.T
    
    for p in range(0, len(L_diag)):
        cgauss_rough[p] = cgauss_rough[p]/L_diag[p]
    
    
    cgauss_rough = np.abs(cgauss_rough)
    
    cgauss_rough_mean = np.dot(cgauss_rough.T, mass.diagonal(0))
    cgauss_rough_mean = cgauss_rough_mean / surface
    
    
    #power model for modulating the roughness
    
    minrough = 0.0005
    maxrough1 = 0.20
    maxrough2 = 5.0 * cgauss_rough_mean[0]
    maxrough = max(maxrough1,maxrough2)
    
    
    cgauss_rough = np.clip(cgauss_rough,minrough,maxrough)
    
#    a = 0.15
    epsilon = minrough
    cgauss_rough_final = (cgauss_rough)**a - (epsilon)**a
    
    return cgauss_rough_final
    
    

def meshEntropy(blade_verts, blade_colors, patchSize = 80):    
    
    pcd = PointCloud()
    
    pcd.points = Vector3dVector(blade_verts)
    
    print(pcd)
    
    # Calculate the KDTree from the provided point cloud
    pcd_tree = KDTreeFlann(pcd)
    

    # for checking if all vertices have been used
    allVerts = np.ones([len(blade_verts),1])
    
   
    
    blade_colors_masked = np.zeros([len(blade_colors),1])
    
    allEntropy = []

    
    # Go through each point
    i=0
    while i<len(blade_verts):
        

        
        # Get the neighbours around each point
        [k_small, idx_small, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], patchSize)
        currNdx = np.array(idx_small)
    
    
        nearestNeighbors_colors = blade_colors[currNdx]
    
    
        #  RGB to intensity for each point
        nearestNeighbors_intensity = 0.2989 * nearestNeighbors_colors[:,0] + 0.5870 * nearestNeighbors_colors[:,1] + 0.1140 * nearestNeighbors_colors[:,2]                           
    
        #  calculate entropy of the current point depending on the colors in area around it
        currEntropy = get_entropy(nearestNeighbors_intensity)
        
        #  give new entropy value to the mask array
        blade_colors_masked[i] = currEntropy
    
        allEntropy.append(currEntropy)                      
                                      
    
        
        allVerts[currNdx,0] = 0
        
        i+=1
        
    

    allEntropy = np.array(allEntropy)
    
    
    return blade_colors_masked
    
# https://www.researchgate.net/publication/235924402_Difference_of_Normals_as_a_Multi-Scale_Operator_in_Unorganized_Point_Clouds
def meshDifferenceOfNormals(blade_verts, blade_norms, bigRad_percent = 0.02):    

    pcd = PointCloud()
    
    
    pcd.points = Vector3dVector(blade_verts)
    
    print(pcd)
    # Calculate the KDTree from the provided point cloud
    pcd_tree = KDTreeFlann(pcd)
    
    
    
    allVerts = np.ones([len(blade_verts),1])
    
    allCheckedNdx = []
    
    allDeltaNorms = []
    allDeltaNorms_magnitude = []
    
    #  Calculate the size of the object in X, Y, Z, get the average size and use that for calculating the big radius as a percentage of it bigRad_percent
    #  The small radius is set as always 10 times smaller than the big radius
    
    minX = np.min( blade_verts[:,0])
    maxX = np.max( blade_verts[:,0])
    
    sizeX = maxX - minX
    
    minY = np.min( blade_verts[:,1])
    maxY = np.max( blade_verts[:,1])
    
    sizeY = maxY - minY
    
    minZ = np.min( blade_verts[:,2])
    maxZ = np.max( blade_verts[:,2])
    
    sizeZ = maxZ - minZ
    
    
    sizeAvg = (sizeX + sizeY + sizeZ)/3
    
    bigRad = sizeAvg*bigRad_percent
    smallRad = bigRad/10
    

    i=0
    # array to contain the DON values
    blade_colors_masked = np.zeros([len(blade_verts),1])
    while i<len(blade_verts):
        
        #  get the points in the area under the small and large radii
        [k_small, idx_small, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], smallRad)
        currNdx = np.array(idx_small)
        
        [k_big, idx_big, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], bigRad)
        currNdx_large = np.array(idx_big)
        
        
        allCheckedNdx.extend(currNdx)

    
        # Get the normals of the points in the small and large radius and calculate their average
        nearestNeighbors_normals = blade_norms[currNdx]
        average_norm_small = nearestNeighbors_normals.mean(axis=0)
    
        allVerts[currNdx,0] = 0
    
        nearestNeighbors_normals_large = blade_norms[currNdx_large]
        average_norm_large = nearestNeighbors_normals_large.mean(axis=0)
        
        
        
        # invert normal if necessary
        if (average_norm_small.dot(average_norm_large) >np.pi/2):
            average_norm_small = -average_norm_small
            
        # Calculate the delta between the average small and large radii
        delta_norm = (average_norm_small - average_norm_large)/2
    
        allDeltaNorms.append(delta_norm)
        # Calculate magnitude
        currNormMag = np.sqrt(delta_norm.dot(delta_norm))
        
        allDeltaNorms_magnitude.append(currNormMag)
        
        blade_colors_masked[i] = currNormMag
        
        
        i+=1
        
        
    
    allDeltaNorms_magnitude= np.array(allDeltaNorms_magnitude)
    
    
    return blade_colors_masked



def meshPointDensityAndColorUniformity(blade_verts, percentFromMaxNeighbour = 0.3):
    pcd = PointCloud()


    pcd.points = Vector3dVector(blade_verts)
    
    print(pcd)
    # Calculate the KDTree from the provided point cloud
    pcd_tree = KDTreeFlann(pcd)
    
    
    # Calculate the size of the object in X, Y, Z, get the average size and use that for calculating the search radius and the change of the different sizes search radii
    
    minX = np.min(blade_verts[:, 0])
    maxX = np.max(blade_verts[:, 0])
    
    sizeX = maxX - minX
    
    minY = np.min( blade_verts[:,1])
    maxY = np.max( blade_verts[:,1])
    
    sizeY = maxY - minY
    
    minZ = np.min( blade_verts[:,2])
    maxZ = np.max( blade_verts[:,2])
    
    sizeZ = maxZ - minZ
    
    
    sizeAvg = (sizeX + sizeY + sizeZ)/3
    # The search radius is 0.5% of the size, while the change of that radius is between 20% bigger and smaller than it
    searchRad = sizeAvg*0.005
    searchRad_delta =searchRad*0.20
    

    
    blade_colors_masked = np.zeros([len(blade_verts),1])
    
    maxNeighbours_all = []
    
    numMaxesForAvg = 20
    
    # Get the maximum possible neighbours for the current mesh/point cloud this is used for calculating how many neighbours are possible
    #  the 0.05 is the step ammount
    for k in np.arange(searchRad-searchRad_delta, searchRad+searchRad_delta, searchRad*0.05):
        
        i=0
        maxNeighbours = 0
        
        currAll_neightbours = np.zeros([1,len(blade_verts)])
        while i<len(blade_verts):
            [k_big, idx_big, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], k)
            currNdx_large = np.array(idx_big)
            
            currAll_neightbours[0,i] = len(currNdx_large)
        
            i+=1
        
        
        maxNeighbours = np.median(currAll_neightbours[0,currAll_neightbours[0,:].argsort()[-numMaxesForAvg:][::-1]])
        print("Maximum neighbours for R=" + str(k) + " is " + str(maxNeighbours) )
        #  the max neighbourhoods for each radius step are calculated for the whole object and used as thresholds
        maxNeighbours_all.append(maxNeighbours)
     
    i=0
    
    # for each point the different radius steps are tested and the amount of neighbours for each are calculated
    #  if it above a certain percentage of the maximum possible then that point gets a heigher rating
    while i<len(blade_verts):
        
        count=0
        for j in np.arange(searchRad-searchRad_delta, searchRad+searchRad_delta,searchRad*0.05):
            [k_big, idx_big, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], j)
            currNdx_large = np.array(idx_big)
            
            numNeighbours = percentFromMaxNeighbour * maxNeighbours_all[count] 
            
            if len(currNdx_large) >numNeighbours:
                              
                blade_colors_masked[i,0] += 1
            count+=1

        i+=1
        
        
    return blade_colors_masked