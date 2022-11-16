import RPi.GPIO as GPIO
import cv2
import jetson.inference
import jetson.utils
import time
import numpy as np

def addEvent(exitTime,condition,index):
    if index==-1:
        #add new element
        if net.GetClassDesc(detection.ClassID)=="Good":
            quality.append(100)
        if net.GetClassDesc(detection.ClassID)=="Break":
            quality.append(0)
        if net.GetClassDesc(detection.ClassID)=="Color":
            quality.append(10)
        if net.GetClassDesc(detection.ClassID)=="Scratch":
            quality.append(30)
        servoActTimes.append(exitTime)
        servoActTimes.sort()
    else:
        #low pass filter
        if net.GetClassDesc(detection.ClassID)=="Good":
            quality[index]=0.9*quality[index]+10
        if net.GetClassDesc(detection.ClassID)=="Break":
            quality[index]=0.9*quality[index]
        if net.GetClassDesc(detection.ClassID)=="Color":
            quality[index]=0.9*quality[index]+1
        if net.GetClassDesc(detection.ClassID)=="Scratch":
            quality[index]=0.9*quality[index]+3
        
        servoActTimes[index]=0.95*servoActTimes[index]+0.05*exitTime



#Servo init 
servoPin1=33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servoPin1,GPIO.OUT,initial=GPIO.HIGH)
servo1=GPIO.PWM(servoPin1,300)

accept=24  #Servo accept tilt degree 
reject=38  #Servo reject tilt degree
neutral=30 #Servo neutral tilt degree
servo1.start(neutral)

#Camera Setting
width=1280
height=720

cam=jetson.utils.gstCamera(width,height,'0')
display=jetson.utils.glDisplay()

#Font setting
font=jetson.utils.cudaFont()

#time keeper
moment = round(time.time())

#Constanta
speed=0.35 #pixel per ms was 0.7
deguchi=1000 #gate out pixel
waitTime=0.7 #time to wait till next tomato
tHold=90 #Threshold > means accept.

#Variable
servoActTimes=list() #time when servo must act
quality=list() #quality of fruit

objToExit=0 #distance object from exit (pixels)
etime=0 #Estimated exit time
actionTime=0 #last action time
sleep=0 #servo sleeptime
decide=0 #0 -> accept, 1->reject

#AI Network
#net=jetson.inference.detectNet('ssd-mobilenet-v2',['--model=/home/arief/Desktop/myDownloads/jetson-inference/python/training/detection/ssd/models/tomatoSurface/ssd-mobilenet.onnx','--input_blob=input_0','--output_cvg=scores','--output-bbox=boxes','--labels=/home/arief/Desktop/myDownloads/jetson-inference/python/training/detection/ssd/models/tomatoSurface/labels.txt'],threshold=0.7)
net=jetson.inference.detectNet('ssd-mobilenet-v2',['--model=/home/arief/Desktop/myDownloads/jetson-inference/python/training/detection/ssd/models/tomatoMoving/ssd-mobilenet.onnx','--input_blob=input_0','--output_cvg=scores','--output-bbox=boxes','--labels=/home/arief/Desktop/myDownloads/jetson-inference/python/training/detection/ssd/models/tomatoMoving/labels.txt'],threshold=0.7)

#main loop
while display.IsOpen():
    
    if len(servoActTimes)>0 and sleep==0:
        if servoActTimes[0]<time.time():
            actionTime=servoActTimes.pop(0)
            qValue=quality.pop(0)
            print("Action: ",qValue," ", actionTime)
            sleep=1
            if qValue>=tHold:
                servo1.ChangeDutyCycle(accept)
            else:
                servo1.ChangeDutyCycle(reject)


    if sleep==1 and actionTime+waitTime<time.time():
        sleep=0
        #change servo position to netral
        servo1.ChangeDutyCycle(neutral)

    #get Frame   
    frame,width,height=cam.CaptureRGBA(zeroCopy=1)
    detections=net.Detect(frame,width,height)
    timeNow=round(time.time(),3)
    for detection in detections:
        #Classify object
        classified=net.GetClassDesc(detection.ClassID)
        #get estimated exit time for object
        if detection.Center[0]<deguchi and detection.Center[1]>200 and detection.Center[1]<520:
            objToExit=deguchi-detection.Center[0]
            etime=timeNow+(round(objToExit/speed/deguchi,3))
            #check if extimated exittime is registered on list?
            if len(servoActTimes)>0:
                count=0
                foundEvent=False
                for servoActTime in servoActTimes:
                    timeGap=abs(etime-servoActTime)
                    print('timeGap:',etime-servoActTime," index:",count)
                    if timeGap<waitTime:
                        addEvent(etime,classified,count)
                        foundEvent=True
                        print("lowPass")
                        break
                    count=count+1

                if foundEvent==False:
                    print("addNew")
                    addEvent(etime,classified,-1)

            else:
                if etime>actionTime+waitTime:
                    print("add new on empty stack")
                    addEvent(etime,classified,-1)
    
    display.RenderOnce(frame,width,height)

#Closing
servo1.stop()
GPIO.cleanup() 