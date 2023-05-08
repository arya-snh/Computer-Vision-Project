import numpy as np
import cv2

cap = cv2.VideoCapture(0)

img1 = cv2.imread('Image1.jpg')
myVid = cv2.VideoCapture('cv_video_1.mp4')

success, imgVideo = myVid.read()

# Resize img1 while maintaining aspect ratio
target_width = 400  # Set the desired width
scale_percent = target_width / img1.shape[1]  # Calculate the scale percentage
target_height = int(img1.shape[0] * scale_percent)  # Calculate the target height

img1 = cv2.resize(img1, (target_width, target_height))

ht,wt,ct = img1.shape
imgVideo = cv2.resize(imgVideo,(wt,ht))

orb = cv2.ORB_create(nfeatures = 1000)

kp1,des1 = orb.detectAndCompute(img1,None)


#img1 = cv2.drawKeypoints(img1,kp1,None)


while True:
  img_tgt = cv2.imread('TargetImage.jpg')


  target_width = 400  # Set the desired width
  scale_percent = target_width / img_tgt.shape[1]  # Calculate the scale percentage
  target_height = int(img_tgt.shape[0] * scale_percent)  # Calculate the target height

  img_tgt = cv2.resize(img_tgt, (target_width, target_height))


  kp2,des2 = orb.detectAndCompute(img_tgt,None)

  #img_tgt = cv2.drawKeypoints(img_tgt,kp2,None)

  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2,k=2)
  good = []

  for m,n in matches:
    if m.distance < 0.75*n.distance:
      good.append(m)

  print(len(good))
  imgFeatures = cv2.drawMatches(img1,kp1,img_tgt,kp2,good,None,flags=2)

  if(len(good)>20):
    srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstpts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    matrix,mask = cv2.findHomography(srcPts,dstpts,cv2.RANSAC,5)
    print(matrix)

    pts = np.float32([[0,0],[0,ht],[wt,ht],[wt,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,matrix)
    img2 = cv2.polylines(img_tgt,[np.int32(dst)],True,(255,0,255),3)

  cv2.imshow('img2',img2)
  cv2.imshow('Image Features',imgFeatures)
  cv2.imshow('Image 1',img1)
  #cv2.imshow('Image Video',imgVideo)
  cv2.imshow('Image tgt',img_tgt)
  cv2.waitKey(0)
  break