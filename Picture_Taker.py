import numpy as np
import cv2

import RaspberryPiCameraStream
    
def main():
    rawCapture = RaspberryPiCameraStream.cameraStreamInit()
    savedImages = 0 
    for savedImages in range(0,10):
        image = RaspberryPiCameraStream.getCameraStream(rawCapture)
        cv2.imshow('h', image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            savedImages += 1
            cv2.imwrite('/home/pi/test/AndromedaVision/CameraImages12-28/Image%d.png' % savedImages, image)
            print 'saved'
        cv2.destroyAllWindows()
main()

