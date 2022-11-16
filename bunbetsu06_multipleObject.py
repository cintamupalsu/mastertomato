import RPi.GPIO as GPIO
import cv2
import jetson.inference
import jetson.utils
import time
import numpy as np



#Servo init 
servoPin1=33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servoPin1,GPIO.OUT,initial=GPIO.HIGH)
servo1=GPIO.PWM(servoPin1,300)

accept=38  #Servo accept tilt degree 
reject=24  #Servo reject tilt degree
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
speed=3.3 #pixel per ms
deguchi=1000 #gate out pixel
waitTime=0.35 #time to wait till next tomato

#Variable
servoActTime=[0,0] #time when servo must act
objToExit=0 #distance object from exit (pixels)
etime=0
sleep=0 #servo sleeptime
decide=0 #0 -> accept, 1->reject

#AI Network
net=jetson.inference.detectNet('ssd-mobilenet-v2',['--model=/home/arief/Desktop/myDownloads/jetson-inference/python/training/detection/ssd/models/tomatoSurface/ssd-mobilenet.onnx','--input_blob=input_0','--output_cvg=scores','--output-bbox=boxes','--labels=/home/arief/Desktop/myDownloads/jetson-inference/python/training/detection/ssd/models/tomatoSurface/labels.txt'],threshold=0.7)

#main loop
while display.IsOpen():
    #print(str(etime)+" "+str(round(time.time(),3))+" "+str(sleep))

    if etime<time.time() and sleep==0 and etime>0:
            sleep=1
            if decide==0:
                servo1.ChangeDutyCycle(accept)
            else:
                servo1.ChangeDutyCycle(reject)

    if sleep==1 and etime+waitTime<time.time():
        sleep=0
        etime=0
        servo1.ChangeDutyCycle(neutral)

    #get Frame   
    frame,width,height=cam.CaptureRGBA(zeroCopy=1)
    detections=net.Detect(frame,width,height)
    timeNow=round(time.time(),3)
    for detection in detections:
        objToExit=deguchi-detection.Center[0]
        if etime==0:
            etime=timeNow+(round(objToExit/speed/deguchi,3))
            if net.GetClassDesc(detection.ClassID)=="Good":
                decide=0
            else:
                decide=1
            #print(str(etime)+" "+str(round(time.time(),3))+" "+str(sleep))
        #servoActTime.extend(timeNow+(objToExit/3.3/1000))
        #print(str(eTime)+" "+str(timeNow)+" "+ str(round(objToExit/3.3/1000,3)))
    
        #net.GetClassDesc(detection.ClassID)
    
    display.RenderOnce(frame,width,height)

#Closing
servo1.stop()
GPIO.cleanup()