# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 19:38:06 2021

@author: nehal
"""

import cv2
import tracker

tracker=tracker.EuclideanDistTracker()

cap=cv2.VideoCapture('highway.mp4')

# Detecting Moving Objects
object_detector=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)



while True:
    ret , frame = cap.read()
    height,width,_=frame.shape
    #print(height,width)
    
    # ROI
    roi = frame[340:720,500:800]
    
    # Object Detection
    mask=object_detector.apply(roi)
    # Eliminating Shadow
    _,mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    detection=[]
    
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Removing small elements by calculating area
        area=cv2.contourArea(cnt)
        if area >  100:
            #cv2.drawContours(roi,[cnt],-1,(0,255,0),2)
            x,y,w,h=cv2.boundingRect(cnt)
            #cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
    
            detection.append([x,y,w,h])
            
    box_ids=tracker.update(detection)
    #print(boxes_ids)
    for box_id in box_ids:
        x,y,w,h, id = box_id
        cv2.putText(roi,str(id),(x,y-15),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
    
    #print(detection)
    cv2.imshow("ROI",roi)
    cv2.imshow("Frame",frame)
    cv2.imshow("Mask",mask)
    
    key=cv2.waitKey(30)
    if key==27:
        break
    
cap.release()
cv2.destroyAllWindows()

