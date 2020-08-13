# Extract Features from 3D meshes for Machine Learning
 Two Python scripts for extracting hand crafted features from 3D meshes and point clouds for Machine Learning. Features have been used as data for Random Forests, AdaBoost, Descision Trees, etc.

# Extract Roughness Features

The code contains the implementation to extract four types of features from meshes:
1. Mesh roughness from Gaussian Curvature, presented in the paper ** Wang, K., Torkhani, F., & Montanvert, A. (2012). A fast roughness-based approach to the assessment of 3D mesh visual quality. Computers & Graphics, 36(7), 808-818.**
2. Roughness feature based on Difference of Normals (DON), presented in the paper ** Ioannou, Y., Taati, B., Harrap, R., & Greenspan, M. (2012, October). Difference of normals as a multi-scale operator in unorganized point clouds. In 2012 Second International Conference on 3D Imaging, Modeling, Processing, Visualization & Transmission (pp. 501-508). IEEE.**
3. A vertex local spacial density, inspired by algorithms like the one presented in the paper ** Rabbani, T., Van Den Heuvel, F., & Vosselmann, G. (2006). Segmentation of point clouds using smoothness constraint. International archives of photogrammetry, remote sensing and spatial information sciences, 36(5), 248-253.**
4. A vertex local intensity entropy, for color segmentation directly on the surface of the mesh/point cloud.

**REQUEREMENTS**:
The codebase requires [open3d](http://www.open3d.org/) for calculating the KDTree and finding the local neighbours of each point/vertex and [vispy](http://vispy.org/) for visualization of larger point clouds and meshes. In addition to that Numpy and Scipy need to be also installed.

**Testing Data**
A reconstructed statue of an angel is given as an example input - containing in separate files - the vertices, normals, faces and colors. Open3D can be used to directly input a .ply file containing the same information.

# Extract Geometrical Features

The code contains geometrical features presented in the work of:
1. **Blomley, R., Jutzi, B., & Weinmann, M. (2016). CLASSIFICATION OF AIRBORNE LASER SCANNING DATA USING GEOMETRIC MULTI-SCALE FEATURES AND DIFFERENT NEIGHBOURHOOD TYPES. ISPRS Annals of Photogrammetry, Remote Sensing & Spatial Information Sciences, 3(3).**
2. **Weinmann, M., Jutzi, B., Hinz, S., & Mallet, C. (2015). Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers. ISPRS Journal of Photogrammetry and Remote Sensing, 105, 286-304.**

These features are separated into three types.

**Local Covariance features**
1. Linearity
2. Planarity
3. Sphericity
4. Omnivariance
5. Anisotropy
6. Eigentropy
7. Sum of Eigenvalues
8. Local Surface Variation

**Geometrical Features**
1. Local Density
2. Farthest Distance
3. Maximum Height
4. Height Std. Dev.

**Statistical Shape Distribution features**
1. Distance between random point and centeroid
2. Distance between two random point in the neighbourhood
3. Square root of the area between three random points
4. Cubic root of the volume between four random points

**REQUEREMENTS**:
The codebase requires [open3d](http://www.open3d.org/) for calculating the KDTree and finding the local neighbours of each point/vertex and [vispy](http://vispy.org/) for visualization of larger point clouds and meshes. In addition to that Numpy and Scipy need to be also installed.

**Testing Data**
A 3D reconstructed sandpaper point cloud is given as an example input - containing in separate files - the vertices and normals. Open3D can be used to directly input a .ply file containing the same information.
