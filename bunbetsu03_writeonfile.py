import cv2
import jetson.inference
import jetson.utils
import time
import numpy as np

#from threading import Thread

#Recording function (failed)
#def FrameRecording(frame):
#    frameBGR=jetson.utils.cudaToNumpy(frame,width,height,4)
#    frameBGR=cv2.cvtColor(frameBGR, cv2.COLOR_RGBA2BGR).astype(np.uint8)
#    outVid.write(frameBGR)

#display width and height
width=1280
height=720

#camera setting
cam=jetson.utils.gstCamera(width,height,'0')
display=jetson.utils.glDisplay()

#font
font=jetson.utils.cudaFont()

#AI Network
net=jetson.inference.detectNet('ssd-mobilenet-v2',['--model=/home/arief/Desktop/myDownloads/jetson-inference/python/training/detection/ssd/models/tomatoSurface/ssd-mobilenet.onnx','--input_blob=input_0','--output_cvg=scores','--output-bbox=boxes','--labels=/home/arief/Desktop/myDownloads/jetson-inference/python/training/detection/ssd/models/tomatoSurface/labels.txt'],threshold=0.7)

#Time keeper
timeMark=time.time()
fpsFilter=0 

#otherVar
countObject=0 #count Object found in screen
center=0

#Video recorder
outVid=cv2.VideoWriter('videos/jetsonAI.avi', cv2.VideoWriter_fourcc(*'XVID'),21,(width,height))

#file txt recorder
with open('IBM/txtfiles/bunbetsu03.txt','w') as file:
    #Main loop
    while display.IsOpen():
        #get Frame
        frame,width,height=cam.CaptureRGBA(zeroCopy=1)
        detections=net.Detect(frame,width,height)
        arrayNum = 0
        for detection in detections:
            arrayNum = arrayNum + 1
            file.write('x'+str(arrayNum) +':'+str(round(detection.Center[0]))+ ',time:'+str(round(time.time(),3))+',name:'+ net.GetClassDesc(detection.ClassID)+ '\n') #Center[0] is x
        """
        if detections:
            #countObject = len(detections)
            #font.OverlayText(frame,width,height,'found '+str(countObject)+' objects',5,5,font.Yellow,font.Blue)
            
            if center<=0:
                center=detections[0].Center[0]
            else:
                center=0.8*center+0.2*detections[0].Center[0]
            file.write(str(round(center,1))+' '+str(time.time())+'\n')
            font.OverlayText(frame,width,height,'x[0]= '+str(round(center,1))+' pix',5,5,font.Yellow,font.Blue)
        else:
            countObject = 0
            center=0
        
        #Video recording
        #frameBGR=jetson.utils.cudaToNumpy(frame,width,height,4)
        #frameBGR=cv2.cvtColor(frameBGR, cv2.COLOR_RGBA2BGR).astype(np.uint8)
        #outVid.write(frameBGR)
        #Threading (Failed)
        #recordingThread=Thread(target=FrameRecording,args=(frame))
        #recordingThread.daemon=True
        #recordingThread.start()
        """
        display.RenderOnce(frame,width,height)
    
    #close recorder txt file.
    file.close()
#close video\
outVid.release()
        

