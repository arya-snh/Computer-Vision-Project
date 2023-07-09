The goal is to develop an augmented reality system that can insert a 3D object onto a flat surface in a video using planar homography in real-time such that its position and orientation align with the specific flat surface. 
Additionally, the projection should update in real-time as the surface changes its position or orientation.

Feature extraction and description computation
Distinctive features are identified in both the reference and target images to locate the object in the target image. 

After identifying features in images, they are assigned descriptors using use ORB (Oriented Fast and Rotated Brief). The descriptors are then transformed into a feature vector, typically in the form of binary strings, to abstract the object for recognition purposes. 

Feature matching
In the matching process, the k-nearest neighbours algorithm (kNN). We used the OpenCV kNN search algorithm to return the two closest descriptors from the reference image for every descriptor from the current target image.


![image](https://github.com/arya-snh/Computer-Vision-Project/assets/114855347/b16d56a7-b4b0-4a26-999e-2748eecfc56f)

