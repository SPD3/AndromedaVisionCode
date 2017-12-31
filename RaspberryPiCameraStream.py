import numpy as np
import cv2
from picamera.array import PiRGBArray
import picamera
import time


m_xResolution = 1024 
m_yResolution = 768
m_nameOfRaspberryPi = '4905pi-2'
m_camera = picamera.PiCamera(resolution = (m_xResolution, m_yResolution))
m_cameraMatrix = np.load('/home/pi/test/AndromedaVision/' + m_nameOfRaspberryPi + "/mtx.npy")
m_distCoeffs = np.load('/home/pi/test/AndromedaVision/' + m_nameOfRaspberryPi + "/dist.npy")
m_fieldOfViewAngle = 62.2

def cameraStreamInit():
    #m_camera.resolution = (m_xResolution, m_yResolution)
    m_camera.framerate = 32
    m_camera.shutter_speed = 800
    m_camera.iso = 100
    m_camera.exposure_mode = 'off'
    m_camera.flash_mode = 'off'
    m_camera.awb_mode = 'off'
    m_camera.drc_strength = 'off'
    m_camera.led = False
    m_camera.awb_gains = 1
    rawCapture = PiRGBArray(m_camera, size=(m_xResolution, m_yResolution))
 
    # allow the camera to warmup
    time.sleep(0.1)
    return rawCapture

def getCameraStream(rawCapture):
    for frame in m_camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        image = frame.array
        rawCapture.truncate(0)
        h,w = image.shape[:2]
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(m_cameraMatrix,m_distCoeffs,(w,h),1,(w,h))
        undistortedImage = cv2.undistort(image, m_cameraMatrix, m_distCoeffs, None, newCameraMtx)
        return undistortedImage
