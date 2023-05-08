import numpy as np
import cv2
import math
import os
from objloader_simple import *
DEFAULT_COLOR = (0, 0, 0)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape[0], model.shape[1]

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


cap = cv2.VideoCapture(0)

img1 = cv2.imread('image.png')
myVid = cv2.VideoCapture('cv_video_1.mp4')
 
detection = False
frameCounter = 0
success, imgVideo = myVid.read()

# Resize img1 while maintaining aspect ratio
target_width = 400  # Set the desired width
scale_percent = target_width / img1.shape[1]  # Calculate the scale percentage
target_height = int(img1.shape[0] * scale_percent)  # Calculate the target height
camera_parameters = np.array([[955.88, 0, 566.63], [0, 958.46, 360.43], [0, 0, 1]])
img1 = cv2.resize(img1, (target_width, target_height))

ht,wt,ct = img1.shape
imgVideo = cv2.resize(imgVideo,(wt,ht))

orb = cv2.ORB_create(nfeatures = 1000)

kp1,des1 = orb.detectAndCompute(img1,None)


img1 = cv2.drawKeypoints(img1,kp1,None)
# cv2.imshow('Image 1',img1)
# cv2.waitKey(0)
total_count = myVid.get(cv2.CAP_PROP_FRAME_COUNT)
obj = OBJ('fox.obj', swapyz=True) 
while True:

  # img_tgt = cv2.imread('TargetImage.jpg')
  ret, img_tgt = cap.read()
  target_width = 400  # Set the desired width
  scale_percent = target_width / img_tgt.shape[1]  # Calculate the scale percentage
  target_height = int(img_tgt.shape[0] * scale_percent)  # Calculate the target height

  img_tgt = cv2.resize(img_tgt, (target_width, target_height))
  img_aug = img_tgt.copy()

  kp2,des2 = orb.detectAndCompute(img_tgt,None)

  #img_tgt = cv2.drawKeypoints(img_tgt,kp2,None)
  if detection==False:
    myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frameCounter = 0
  else:
      # if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
      #   myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
      #   frameCounter = 0
      success, imgVideo = myVid.read()
      imgVideo = cv2.resize(imgVideo,(wt,ht))

  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2,k=2)
  good = []

  for m,n in matches:
    if m.distance < 0.75*n.distance:
      good.append(m)

  print(len(good))
  imgFeatures = cv2.drawMatches(img1,kp1,img_tgt,kp2,good,None,flags=2)

  if(len(good)>2):
    detection = True
    srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstpts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    matrix,mask = cv2.findHomography(srcPts,dstpts,cv2.RANSAC,5)
    print(matrix)

    pts = np.float32([[0,0],[0,ht],[wt,ht],[wt,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,matrix)
    img2 = cv2.polylines(img_tgt,[np.int32(dst)],True,(255,0,255),3)

    imgWarp = cv2.warpPerspective(imgVideo, matrix, (img_tgt.shape[1], img_tgt.shape[0]))
    maskNew = np.zeros((img_tgt.shape[0], img_tgt.shape[1]), np.uint8)
    cv2.fillPoly(maskNew,  [np.int32(dst)], (255,255,255))
    maskInv = cv2.bitwise_not(maskNew)
    img_aug = cv2.bitwise_and(img_aug, img_aug, mask = maskInv)
    img_aug = cv2.bitwise_or(imgWarp, img_aug )

  projection = projection_matrix(camera_parameters, matrix)  
  # project cube or model
  print(img1.shape)
  frame = render(img2, obj, projection, img1, False)
  cv2.imshow('frame',frame)
  # cv2.imshow('imgWarp',imgWarp)
  # cv2.imshow('img2',img2)
  # cv2.imshow('Image Features',imgFeatures)
  # cv2.imshow('Image 1',img1)
  # cv2.imshow('Image Video',imgVideo)
  # cv2.imshow('Image tgt',img_tgt)
  cv2.waitKey(0)
  frameCounter+=1
  # break