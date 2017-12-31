import numpy as np
import cv2

import Filtration
import RaspberryPiCameraStream

def main():
    rawCapture = RaspberryPiCameraStream.cameraStreamInit()
    
    while True:
        image = RaspberryPiCameraStream.getCameraStream(rawCapture)
        cv2.imshow('LiveFeed: ' , image)
        cv2.waitKey(10)
        
        foundTarget, target, filTrationType = Filtration.findTarget(image)
        print foundTarget
        if foundTarget:
            angleToTurn = (RaspberryPiCameraStream.m_fieldOfViewAngle/RaspberryPiCameraStream.m_xResolution)*(target[0][0]- RaspberryPiCameraStream.m_xResolution/2)
        else:
            continue

main()
            

