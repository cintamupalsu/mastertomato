import cv2
import jetson.inference
import jetson.utils
import time
import numpy as np

#display width and height
width=1280
height=720

#camera setting
flip=2
camSet='nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(width)+', height='+str(height)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam1=cv2.VideoCapture(camSet)
display=jetson.utils.glDisplay()

#AI net 
net=jetson.inference.detectNet('ssd-mobilenet-v2',['--model=/home/arief/Desktop/myDownloads/jetson-inference/python/training/detection/ssd/models/tomatoSurface/ssd-mobilenet.onnx','--input_blob=input_0','--output_cvg=scores','--output-bbox=boxes','--labels=/home/arief/Desktop/myDownloads/jetson-inference/python/training/detection/ssd/models/tomatoSurface/labels.txt'])

#Time keeper
timeMark=time.time()
fpsFilter=0
font=cv2.FONT_HERSHEY_SIMPLEX

#HSV setting
hue1L=0 #hue 1 lower
hue1U=50 #hue 1 upper
hue2L=165 #hue 2 lower
hue2U=179 #hue 2 upper
satL=90 #sat lower
satU=255 #sat upper
valL=0 #val lower
valU=255 #val upper
#Covert HSV setting to array lower bound and upper bound
l_b1=np.array([hue1L,satL,valL])
u_b1=np.array([hue1U,satU,valU])
l_b2=np.array([hue2L,satL,valL])
u_b2=np.array([hue2U,satU,valU])

#Constanta
minObjectArea=5000
roiScale=3

#Main loop
while True:
    #Get image from camera
    ret,frame=cam1.read()
    
    #Convert frame to HSV for masking
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #Create Foreground mask
    FGMask1=cv2.inRange(hsv,l_b1,u_b1)
    FGMask2=cv2.inRange(hsv,l_b2,u_b2)
    FGMaskComp=cv2.add(FGMask1,FGMask2)

    #Temp MaskResult show
    #cv2.imshow('FGMaskComp', FGMaskComp)
    #cv2.moveWindow('FGMaskComp', 0,0)

    #Region of Interest (ROI)
    contours,_=cv2.findContours(FGMaskComp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        (x,y,w,h)=cv2.boundingRect(cnt)
        if area>=minObjectArea:

            #Crop ROI
            if w>0 and h>0:
                roi=frame[y:y+h,x:x+w]
                roi=cv2.resize(roi,(0,0),fx=roiScale,fy=roiScale)
                #cv2.imshow('recCam', roi)
                #cv2.moveWindow('recCam',0,0)
            
                #Convert image to jetson inference readable format (RGBA)
                img=cv2.cvtColor(roi,cv2.COLOR_BGR2RGBA).astype(np.float32)
                img=jetson.utils.cudaFromNumpy(img)

                #Detect object
                detections=net.Detect(img,w*roiScale,h*roiScale)
                if detections:
                    item=net.GetClassDesc(detections[0].ClassID)
                    cv2.putText(frame,item,(x,y-6),font,.75,(0,255,255),2)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)


    #Display the result 
    cv2.imshow('recCam', frame)
    cv2.moveWindow('recCam',0,0)

    #Break the loop
    if cv2.waitKey(1)==ord('q'):
        break

#Closing
cam1.release()
cv2.destroyAllWindows()