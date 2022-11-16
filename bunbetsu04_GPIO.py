import RPi.GPIO as GPIO
import cv2
import jetson.inference
import jetson.utils
import time
import numpy as np
from random import seed
from random import randint


#Servo init 
servoPin1=33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servoPin1,GPIO.OUT,initial=GPIO.HIGH)
servo1=GPIO.PWM(servoPin1,300)

accept=38  #Servo accept tilt degree 
reject=24 #Servo reject tilt degree
neutral=30
servo1.start(neutral)
#servo1.start(accept)
#servo1.start(reject)

#Camera Setting
width=1280
height=720
cam=jetson.utils.gstCamera(width,height,'0')
display=jetson.utils.glDisplay()

#Font setting
font=jetson.utils.cudaFont()

#time keeper
moment = round(time.time())

#randomize
seed(1)
rndNum=randint(0,1)

#Main loop
while display.IsOpen():
    if round(time.time())-moment>2:
        if rndNum>0:
            servo1.ChangeDutyCycle(accept)
        else:
            servo1.ChangeDutyCycle(reject)
        moment = round(time.time())
        rndNum=randint(0,1)
    frame,width,height=cam.CaptureRGBA(zeroCopy=1)
    display.RenderOnce(frame,width,height)

#Closing
servo1.stop()
GPIO.cleanup()