import numpy as np
import cv2
import socket
import struct
import datetime

import Filtration
import RaspberryPiCameraStream

UDP_IP = "roborio-4905-frc.local" #"10.49.5.77" #
UDP_PORT = 4445

def main():
    rawCapture = RaspberryPiCameraStream.cameraStreamInit()
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    x = 0
    deltaTimeTotal = 0
    getImageDeltaTimeTotal = 0
    filtrationDeltaTimeTotal =0
    sendOutDateDeltaTimeTotal =0
    while x < 50:
        
        startTime = datetime.datetime.now()
        
        image = RaspberryPiCameraStream.getCameraStream(rawCapture)
        gotImageTime = datetime.datetime.now()
        #cv2.imshow('LiveFeed: ' , image)
        #cv2.waitKey(10)
        
        foundTarget, target, filTrationType = Filtration.findTarget(image)
        
        if foundTarget:
            angleToTurn = (RaspberryPiCameraStream.m_fieldOfViewAngle/RaspberryPiCameraStream.m_xResolution)*((target[0][0] + target[0][2]/2)- RaspberryPiCameraStream.m_xResolution/2)
        else:
            angleToTurn = 0.0
        
        print "angleToTurn: " , angleToTurn
        gotAngleToTurnTime = datetime.datetime.now()
        ba = struct.pack("!d", angleToTurn)
        
        result = sock.sendto(ba,(UDP_IP, UDP_PORT))
        endTime = datetime.datetime.now()
        
        deltaTime = (endTime - startTime).total_seconds()
        getImageDeltaTime = (gotImageTime - startTime).total_seconds()
        filtrationDeltaTime = (gotAngleToTurnTime - gotImageTime).total_seconds()
        sendOutDateDeltaTime = (endTime - gotAngleToTurnTime).total_seconds()

        deltaTimeTotal +=deltaTime
        getImageDeltaTimeTotal += getImageDeltaTime
        filtrationDeltaTimeTotal += filtrationDeltaTime
        sendOutDateDeltaTimeTotal += sendOutDateDeltaTime
        
        x+=1 
     
    print 'totalDeltaTimeAverage: ' , deltaTimeTotal/x
    print 'getImageDeltatTimeAverage: ' , getImageDeltaTimeTotal/x
    print 'filtrationDeltaTimeAverage: ' , filtrationDeltaTimeTotal/x
    print 'sendOutDateDeltaTimeAverage: ' , sendOutDateDeltaTimeTotal/x
        

main()
            

