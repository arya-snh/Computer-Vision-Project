## AR/VR object insertion in a plane using planar homographies

### Feature extraction and description computation </h2>
Distinctive features are identified in both the reference and target images to locate the object in the target image. 

After identifying features in images, they are assigned descriptors using use ORB (Oriented Fast and Rotated Brief). The descriptors are then transformed into a feature vector, typically in the form of binary strings, to abstract the object for recognition purposes. 

In the matching process, the k-nearest neighbours algorithm (kNN) has been used.

![image](https://github.com/arya-snh/Computer-Vision-Project/assets/114855347/16423d03-6290-4abb-8849-8808e6fc0896) </br>
Fig: Feature matching

### Homography estimation

Now the task is to find a transformation matrix, i.e. homography that maps points from the surface plane to the image plane. RANSAC is used for robust parameter estimation.

![image](https://github.com/arya-snh/Computer-Vision-Project/assets/114855347/b994ce0e-2529-4afc-80ab-6c2afb6aaaa5) </br>
Fig: RANSAC Algorithm </br>

![image](https://github.com/arya-snh/Computer-Vision-Project/assets/114855347/d82e66d6-5166-405f-bbf2-727afd44ba12) </br>
Fig: Object bounding box using estimated homography

The homography is extended for any 3D point from reference surface coordinate to target image.

### Result
The following augmented reality system that can insert a 3D object onto a flat surface in a video using planar homography in real-time such that its position and orientation align with the specific flat surface. </br>
Additionally, the projection should update in real-time as the surface changes its position or orientation.

![image](https://github.com/arya-snh/Computer-Vision-Project/assets/114855347/47b475f1-08fa-4985-95db-a99a3cd1ef5c) </br>





