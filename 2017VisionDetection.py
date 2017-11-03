#import pkg_resources
#print pkg_resources.get_distribution('picamera').version
#Testing
import cv2
import numpy as np
import os
import math
from picamera.array import PiRGBArray
import picamera
import time
import sys
from networktables import NetworkTables
import logging
import collections
import ctypes

with open('/home/pi/Desktop/NameOfRaspberryPi') as f:
    m_nameOfRaspberryPi = f.read()
    
with open('/home/pi/test/AndromedaVision/' + m_nameOfRaspberryPi + "/CameraType") as f:
    m_typeOfCamera = f.read()

#Memory Variables
m_shortTermMemory = collections.deque()
m_secondsToSaveMemory = 15
m_microsecondsToSaveMemory = m_secondsToSaveMemory*1000000
m_libc = ctypes.CDLL("libc.so.6") 

#interensic paramaters
m_xResolution = 1024 #2656 
m_yResolution = 768 #1328
m_cameraCalibrationData = np.load('/home/pi/test/AndromedaVision/CameraCalibrationData.npz')

m_cameraMatrix = np.load('/home/pi/test/AndromedaVision/' + m_nameOfRaspberryPi + "/mtx.npy")
m_distCoeffs = np.load('/home/pi/test/AndromedaVision/' + m_nameOfRaspberryPi + "/dist.npy")
#print 'm_cameraMatrix', m_cameraMatrix
#print 'm_distCoeffs', m_distCoeffs
#print m_cameraMatrix
#print np.load('/home/pi/Desktop/mtx.npy')
#print m_distCoeffs
#print np.load('/home/pi/Desktop/dist.npy')



#m_cameraMatrix = np.matrix([[  4.81899392e+03,   0.00000000e+00,   6.93806251e+02],
 #[  0.00000000e+00,   4.93175109e+03 ,  6.23951896e+02],
 #[  0.00000000e+00  , 0.00000000e+00  , 1.00000000e+00]])
#m_distCoeffs = np.matrix([[  1.49221759e-02 ,  4.16414703e+00,  -2.27562975e-02 , -5.36383565e-02,-1.60380458e+01]])
m_centerXOfImage = m_cameraMatrix[0,2]#m_xResolution/2# #Need to load in actual Numbers from Camera Calibration
m_centerYOfImage = m_cameraMatrix[1,2]# m_yResolution/2# #Need to load in actual Numbers from Camera Calibration
m_focalLengthOfCameraX = m_cameraMatrix[0,0] #Need to load in actual Numbers from Camera Calibration
m_focalLengthOfCameraY = m_cameraMatrix[1,1] #Need to load in actual Numbers from Camera Calibration
#m_horizonLine = 0.9 * m_yResolution # #Need to get actual number from camera

#field parameters
m_heightOfHighGoalTarget = 10.0 #Need to get actual number from manual
m_heightOfLiftTarget = 15.75 #Actual Number From manual
m_widthOfLift = 8.25 #Actual number from manual; Top Left corner of retroReflective to Top right Corner Of RetroReflective
m_widthOfRetroReflectiveToLift = m_widthOfLift/2
objPoints = np.matrix([[-5.125,0.0,15.75],[-3.125,0.0,10.75],[-5.125,0.0,10.75],[-3.125,0.0,15.75],[3.125,0.0,15.75],[5.125,0.0,10.75],
                       [3.125,0.0,10.75],[5.125,0.0,15.75]]) #HARD CODE IN THESE VALUES

#m_degreesAngleOfCamera = 18 #16.65 #+ (0.0400313438911 *(180/math.pi))#actual number from Robot
#print 'm_degreesAngleOfCamera ', m_degreesAngleOfCamera




#m_camera = picamera.PiCamera(resolution = (m_xResolution, m_yResolution))


#print m_centerXOfImage, "and", m_centerYOfImage
#print m_xResolution, "by", m_yResolution

#print 'm_typeOfCamera', m_typeOfCamera
m_camera = picamera.PiCamera(resolution = (m_xResolution, m_yResolution))

#Found at learnopencv.com
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0],sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0],sy)
        z = 0
    return np.array([x,y,z])

m_RCamera = np.load('/home/pi/test/AndromedaVision/' + m_nameOfRaspberryPi + "/R.npy")
m_tvecCamera = np.load('/home/pi/test/AndromedaVision/' + m_nameOfRaspberryPi + "/tvec.npy")
#print 'm_RCamera', m_RCamera
#print 'm_tvecCamera', m_tvecCamera
m_eulerAngles = rotationMatrixToEulerAngles(m_RCamera)
m_radiansAngleofCamera = math.pi/2 - m_eulerAngles[0]#(m_degreesAngleOfCamera * (math.pi/180))# - 0.0400313438911
inverseR = np.linalg.inv(m_RCamera)
m_eulerAngles = rotationMatrixToEulerAngles(inverseR)
#print "m_eulerAngles", m_eulerAngles

m_robotTvec = -(inverseR.dot(m_tvecCamera))
m_heightOfCamera = m_robotTvec[2]

#print 'real Tvec: ', m_robotTvec
#print 'm_heightOfCamera', m_heightOfCamera
#print 'm_radiansAngleofCamera', m_radiansAngleofCamera

#extrensic parameters

#offset parameteres
m_lateralRightOffsetOfCamera = m_robotTvec[0] #Need to get actual number from Robot
m_forwardOffsetOfCamera = m_robotTvec[1] #Need to get actual number from Robot
m_lateralRightOffsetOfShooter = 0.0 #Need to get actual number from Robot
m_forwardOffsetOfShooter = 0.0 #Need to get actual number from Robot
m_lateralRightOffsetOfGearPlacer = 0.0 #Need to get actual number from Robot
m_forwardOffsetOfGearPlacer = 0.0 #Need to get actual number from Robot
m_rightOffsetOfGearPlacerFromCamera = 0.0
m_forwardOffsetOfGearPlacerFromCamera = 0.0
m_heightOfHighGoalTargetFromCamera = m_heightOfHighGoalTarget - m_heightOfCamera
m_heightOfLiftTargetFromCamera = m_heightOfLiftTarget - m_heightOfCamera

def cameraStreamInit():
    #m_camera.resolution = (m_xResolution, m_yResolution)
    m_camera.framerate = 10

    m_camera.shutter_speed = 1000

    m_camera.iso = 100
    m_camera.exposure_mode = 'off'
    m_camera.flash_mode = 'off'
    m_camera.awb_mode = 'off'
    m_camera.drc_strength = 'off'
    m_camera.led = False
    m_camera.awb_gains = 1
    #m_camera.MAX_RESOLUTION
    rawCapture = PiRGBArray(m_camera, size=(m_xResolution, m_yResolution))
 
    # allow the camera to warmup
    time.sleep(2)
    return rawCapture

def getRobotTimeStamp(networkTable):
    return networkTable.getNumber("RobotTimestamp", 0.0)

def getCameraStream(rawCapture, networkTable):
    for frame in m_camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        robotTimestamp = getRobotTimeStamp(networkTable)
        timestamp2 = m_camera.timestamp
        image = frame.array
        rawCapture.truncate(0)
        h,w = image.shape[:2]
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(m_cameraMatrix,m_distCoeffs,(w,h),1,(w,h))
        #print 'undistorting'
        undistortedImage = cv2.undistort(image, m_cameraMatrix, m_distCoeffs, None, newCameraMtx)
        #print 'undistorted'
        #small = cv2.resize(image, (0,0), fx = 0.45, fy = 0.45)
        ##cv2.imshow('h', image)
        #cv2.imwrite("/home/pi/Pictures/test.png", image)

        #cv2.waitKey(0)>>>>>>> fb0b1311040ab4c861f94eae3a497421292d0ada
        #cv2.destroyAllWindows()
        return robotTimestamp,undistortedImage, timestamp2
    
def null(x):
    pass

def setupImageWindow():
    #cv2.namedWindow("Original Image")
 #   cv2.namedWindow("Processed Image")
  #  cv2.createTrackbar('deltaX', 'Processed Image',0,10,null)
   # cv2.createTrackbar('lowDeltaYLimit', 'Processed Image',0,100,null)
    #cv2.createTrackbar('highDeltaYLimit', 'Processed Image',0,100,null)
    #cv2.createTrackbar('maxWidth', 'Processed Image',0,500,null)
    #cv2.createTrackbar('maxS', 'Processed Image',0,255,null)
    #cv2.createTrackbar('maxV', 'Processed Image',0,255,null)
    pass
    
def findLiftTarget(img):
    #Runs all the filtiration methods to find the Upper High Goal Target
    correctColorImage = filterColors(img,59,150,5,63,255,75)#(img,55,250,10,60,255,65)
    #cv2.imshow('Processed Image', correctColorImage)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    preparedImage = prepareImage(correctColorImage)    
    copy = preparedImage.copy() #need to do this because the findContours function alters the source image
    correctNumberOfContoursList = filterContours(copy,4)
    #print 'correctNumberOfContoursList: ',len(correctNumberOfContoursList)
    correctSizeList = filterSize(correctNumberOfContoursList,26, 2000,13,2000)
    drawBoundingBoxes(img, correctSizeList)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #print 'correctSizeList: ',len(correctSizeList)
    drawBoundingBoxes(img, correctSizeList)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    correctBlack2WhiteRatioList = filterBlack2WhiteRatio(correctSizeList, preparedImage,-1,0.4)
    #print 'correctBlack2WhiteRatioList: ',len(correctBlack2WhiteRatioList)
    drawBoundingBoxes(img, correctBlack2WhiteRatioList)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    correctLengthToWidthRatioList = filterLength2WidthRatio(correctBlack2WhiteRatioList,0.2,0.6)

    
    #print 'correctLengthToWidthRatioList: ',len(correctLengthToWidthRatioList)
    #drawBoundingBoxes(img, correctLengthToWidthRatioList)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    
    #correctDistanceBetweenTargetsList = filterByOtherTargetLift(correctBlack2WhiteRatioList, 4.4, 25, 30)

    #print 'correctDistanceBetweenTargetsList: ',len(correctDistanceBetweenTargetsList)
    
        
    if len(correctLengthToWidthRatioList) != 2 and len(correctLengthToWidthRatioList) != 0:
        conjoinedBloblist = conjoinAnyBlobs(correctSizeList,0.5)
        betterConjoinedBloblist = []

        #print 'conjoinedBloblist', conjoinedBloblist
        for conjoinedBlob in conjoinedBloblist:
            #print 'len(conjoinedBlob): ',len(conjoinedBlob)
            if len(conjoinedBlob) == 4:
                betterConjoinedBloblist = betterConjoinedBloblist + [conjoinedBlob]
        if len(betterConjoinedBloblist) != 0:

            for conjoinedBlob in betterConjoinedBloblist:
                betterFilteredList = correctLengthToWidthRatioList + [conjoinedBlob]
                #print "adding: ", conjoinedBlob
        else:
            betterFilteredList = correctLengthToWidthRatioList

            #print "here"
        #print 'len(betterFilteredList): ', len(betterFilteredList)
        #print '[betterFilteredList]: ', [betterFilteredList]
        correctSizeList = filterSize(betterFilteredList,10, 2000,10,2000)
        correctBlack2WhiteRatioList = filterBlack2WhiteRatio(correctSizeList, preparedImage,-1,0.5)
        correctLengthToWidthRatioList = filterLength2WidthRatio(correctBlack2WhiteRatioList,0.2,0.6)
        betterFilteredList = filterByOtherTargetLift(correctLengthToWidthRatioList,5,0.2,0.5)
        #print '1'
        #print 'final result: ', len(betterFilteredList)

        drawBoundingBoxes(img, betterFilteredList)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if len(betterFilteredList) != 2 and len(correctLengthToWidthRatioList) == 1:
            return True, correctLengthToWidthRatioList
        
        return len(betterFilteredList) == 2, betterFilteredList
        
    if len(correctLengthToWidthRatioList) == 2 :
        """firstBoundingBox = correctLengthToWidthRatioList[0]
        secondBoundingBox = correctLengthToWidthRatioList[1]
        #drawBoundingBox(img, firstBoundingBox)
        firstX, firstY, firstWidth, firstHeight = firstBoundingBox
        secondX, secondY, secondWidth, secondHeight = secondBoundingBox
        if firstHeight > secondHeight:
            ret, conjoinedBlob = checkForConjoiningBlobs(secondBoundingBox,correctNumberOfContoursList, 0.5)

            #print 'conjoinedBlob: ', conjoinedBlob
            if ret:
                filteredList = [conjoinedBlob, firstBoundingBox]
                
            else:
                filteredList = correctLengthToWidthRatioList
                
        else:
            ret, conjoinedBlob = checkForConjoiningBlobs(firstBoundingBox, correctSizeList, 0.5)
            print 'ret', ret
            print 'conjoinedBlob: ', conjoinedBlob
            
            if ret:
                filteredList = [conjoinedBlob, secondBoundingBox]
                
            else:
                filteredList = correctLengthToWidthRatioList

        #print 'filteredList 1: ', filteredList
        drawBoundingBoxes(img, filteredList)
        #cv2.waitKey(0)

        #cv2.destroyAllWindows()
        filteredList = filterByOtherTargetLift(filteredList, 5, 0.2, 0.5)

        #print 'filteredList 2: ', filteredList
        #print
        #print 'filteredList: ', filteredList
        #for box in filteredList:
         #   #print box
          
          #  drawBoundingBoxes(img, filteredList)
           # #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        
        if len(filteredList) == 2:

            #print 'YES final result: ', len(filteredList)
            return True, filteredList"""
        return True, correctLengthToWidthRatioList
    #print 'final result: 0'
    return False, correctBlack2WhiteRatioList
    
    
    
def findHighGoalTarget(img):
     
    #Runs all the filtiration methods to find the Upper High Goal Target
    correctColorImage = filterColors(img,58,100,5,63,255,190)
    ##cv2.imshow('correctColorImage', correctColorImage)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    preparedImage = prepareImage(correctColorImage)    
    copy = preparedImage.copy() #need to do this because the findContours function alters the source image
    correctNumberOfContoursList = filterContours(copy,4)

    #print 'len(correctNumberOfContoursList)', len(correctNumberOfContoursList)
    drawBoundingBoxes(img, correctNumberOfContoursList)
    #cv2.waitKey()
    correctSizeList = filterSize(correctNumberOfContoursList,10,500,10,500)
    #print 'len(correctSizeList)', len(correctSizeList)
    drawBoundingBoxes(img, correctSizeList)
    #cv2.waitKey()
    correctWidth = filterLength2WidthRatio(correctSizeList, 2.5, 3.5)
    #print 'len(correctWidth)', len(correctWidth)
    drawBoundingBoxes(img, correctWidth)
    #cv2.waitKey()
    correctBlack2WhiteRatioList = filterBlack2WhiteRatio(correctWidth, preparedImage,0,3)
    #print 'len(correctBlack2WhiteRatioList)', len(correctBlack2WhiteRatioList)
    drawBoundingBoxes(img, correctBlack2WhiteRatioList)
    correctFilterByOtherTargetsList = filterByOtherTargetHighGoal(correctBlack2WhiteRatioList,2,20,2)
    #print 'correctFilterByOtherTargetsList', len(correctFilterByOtherTargetsList)
    #cv2.waitKey()
    #correctTopHalfBlack2WhiteRatioList = filterTopHalfBlack2WhiteRatio(correctBlack2WhiteRatioList, preparedImage,1,4)

    #print len(correctTopHalfBlack2WhiteRatioList)
    #drawBoundingBoxes(img, correctTopHalfBlack2WhiteRatioList)
    
    #correctLeftHalfBlack2WhiteRatioList = filterLeftHalfBlack2WhiteRatio(correctTopHalfBlack2WhiteRatioList, preparedImage,0,10)
    #print len(correctLeftHalfBlack2WhiteRatioList)
    #drawBoundingBoxes(img, correctLeftHalfBlack2WhiteRatioList)
    #while True:
     #   minRatio = cv2.getTrackbarPos('minRatio','Processed Image')
      #  maxRatio = cv2.getTrackbarPos('maxRatio','Processed Image')
        #minV = cv2.getTrackbarPos('minV','Processed Image')
        #maxH = cv2.getTrackbarPos('maxH','Processed Image')
        #maxS = cv2.getTrackbarPos('maxS','Processed Image')
        #maxV = cv2.getTrackbarPos('maxV','Processed Image')
        
       # correctLeftHalfBlack2WhiteRatioList = filterLeftHalfBlack2WhiteRatio(correctTopHalfBlack2WhiteRatioList, preparedImage,minRatio,maxRatio)
        #drawBoundingBoxes(preparedImage, correctLeftHalfBlack2WhiteRatioList)
        
        
        #key = #cv2.waitKey(0)
#        if key == ord('q'): # quit
 #           return None
  #      elif key == ord('g'): # good
   #         break
        # Try again on any other key

    #print
#    #print minRatio
 #   #print maxRatio
  #  #print 
   # #print 
    #print 
#    #print 
 #   #print
    #correctDistanceBetweenTargets = filterByDistanceBetweenTargets(correctBlack2WhiteRatioList)
    #print len(correctDistanceBetweenTargets)
    #drawBoundingBoxes(img, correctDistanceBetweenTargets)
    #print
    #distanceUShapeIsFromTarget = getDistanceUShapeIsFromTarget(correctTemplateMatchList)
    filteredList = correctFilterByOtherTargetsList
    return len(filteredList) == 1, filteredList
    
def prepareImage(image):
    #Cancels out very small bits of noice by blurring the image and then eroding it
    erodedImage = cv2.erode(image,(3,3))
    erodedImage = cv2.erode(erodedImage,(3,3))
    #erodedImage = cv2.erode(erodedImage,(3,3))
    #erodedImage = cv2.erode(erodedImage,(3,3))
    #erodedImage = cv2.erode(image,(3,3))
    
    gaussianBlurImage = cv2.GaussianBlur(erodedImage,(3,3),1.6)

    return gaussianBlurImage


def filterColors(image,minH,minS,minV,maxH,maxS,maxV):
    #Filters out all colors but green; Returns color filtered image
    HSVImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSVImg,(minH,minS,minV),(maxH,maxS,maxV))

    return mask

def filterContours(image, numberOfContours):
    #Filters out all "Blobs" with less than "numberOfContours" contours 
    #Returns BOUNDING BOXES of "Blobs" having over 8 contours
    img3,contours,hierarchy = cv2.findContours(image, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    goodBoundingBoxes = []
    for box in contours:
        if len(box)>= numberOfContours:
            goodBoundingBoxes = goodBoundingBoxes + [cv2.boundingRect(box)]
    return goodBoundingBoxes
    #Returns BOUNDING BOXES!!!!

def filterSize(goodBoundingBoxes, minHeightSize, maxHeightSize, minWidthSize, maxWidthSize):
    #Filters out "Blobs" that are way too big or way too small
    betterBoundingBoxes = []
    for box in goodBoundingBoxes:
        width =  box[2]
        height =  box[3]
        if minHeightSize < height < maxHeightSize and minWidthSize < width < maxWidthSize:
            betterBoundingBoxes = betterBoundingBoxes + [box]
    return betterBoundingBoxes

def filterLength2WidthRatio(goodBoundingBoxes, lowLengthToWidthRatio, highLengthToWidthRatio):
    #Filters out all "Blobs" with length to width ratios not between lowLengthToWidthRatio and highLengthToWidthRatio
    betterBoundingBoxes = []          
    for box in goodBoundingBoxes:
        width =  box[2]
        height =  box[3]

        #print 'lowLengthToWidthRatio < (width + 0.0)/ (height+ 0.0) < highLengthToWidthRatio', (width + 0.0)/ (height+ 0.0)
        if lowLengthToWidthRatio < (width + 0.0)/ (height+ 0.0) < highLengthToWidthRatio:
            betterBoundingBoxes = betterBoundingBoxes +  [box]
    return betterBoundingBoxes

def filterBlack2WhiteRatio(goodBoundingBoxes, image, blackToWhiteRatioMin, blackToWhiteRatioMax):
    #Filters out all "Blobs" that do not have a ratio of white to black pixels between blackToWhiteRatioMin - blackToWhiteRatioMax 
    betterBoundingBoxes = []
    for box in goodBoundingBoxes:
        x,y,width,height = box
        tempImage = image[y+height/2:y+height, x:x+width]
        
        numberOfWhitePixels = cv2.countNonZero(tempImage)

        #print 'box', box
        if blackToWhiteRatioMin < ((width*(height/2) - numberOfWhitePixels+ 0.0))/(numberOfWhitePixels) < blackToWhiteRatioMax:#number of black pixels for every white pixel
            betterBoundingBoxes = betterBoundingBoxes + [box]
            #print "the good one: ", ((width*(height/2) - numberOfWhitePixels+ 0.0))/(numberOfWhitePixels)
        else:
            #print 'box', box
            #print "the bad ones: ", ((width*(height/2) - numberOfWhitePixels+ 0.0))/(numberOfWhitePixels)
            ##cv2.imshow('temp Image',tempImage)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            pass
    return betterBoundingBoxes

def filterTopHalfBlack2WhiteRatio(goodBoundingBoxes, image, blackToWhiteRatioMin, blackToWhiteRatioMax):
    #Filters out all "Blobs" that do not have a ratio of white to black pixels between blackToWhiteRatioMin and blackToWhiteRatioMax in the top half of the "Blob" this eliminates upside down and sideways U-shapes
    betterBoundingBoxes = []
    for box in goodBoundingBoxes:
        x,y,width,height = box
        tempImage = image[y:y+height/2, x:x+width]
        numberOfWhitePixels = cv2.countNonZero(tempImage)
        if blackToWhiteRatioMin < ((width*height - numberOfWhitePixels+ 0.0))/(numberOfWhitePixels) < blackToWhiteRatioMax:#number of black pixels for every white pixel
            betterBoundingBoxes = betterBoundingBoxes + [box]
        
    return betterBoundingBoxes

def filterLeftHalfBlack2WhiteRatio(goodBoundingBoxes, image, blackToWhiteRatioMin, blackToWhiteRatioMax):
    #Filters out all "Blobs" that do not have a ratio of white to black pixels between blackToWhiteRatioMin and blackToWhiteRatioMax in the left half of the "Blob" this eliminates upside down and sideways U-shapes
    betterBoundingBoxes = []
    for box in goodBoundingBoxes:
        x,y,width,height = box
        tempImage = image[y:y+height, x:x+width/2]
        numberOfWhitePixels = cv2.countNonZero(tempImage)
        numberOfWhitePixels = cv2.countNonZero(tempImage)
        if blackToWhiteRatioMin < ((width*height - numberOfWhitePixels+ 0.0))/(numberOfWhitePixels) < blackToWhiteRatioMax:#number of black pixels for every white pixel
            betterBoundingBoxes = betterBoundingBoxes + [box]
    return betterBoundingBoxes


def filterByOtherTargetHighGoal(goodBoundingBoxes, yOffsetRatio, xOffsetDelta, heightOffsetRatio):
    #filterByDistanceBetweenTargetsHighGoal(0.5,2,)
    if len(goodBoundingBoxes) == 1:
        return goodBoundingBoxes
    
    betterBoundingBoxes = []
    for box in goodBoundingBoxes:
        x,y,width,height = box
        for secondBox in goodBoundingBoxes:
            if box == secondBox:
                continue
            secondX,secondY,secondWidth,secondHeight = secondBox
            yDifference = height*yOffsetRatio
            
            if 0.8*yDifference < secondY - y < 1.2*yDifference :

                #print "It passes the Y test"
                if secondX - xOffsetDelta < x <secondX + height + xOffsetDelta:
                    #print "It passed the first X test"
                
                    if (0.8*heightOffsetRatio < height/secondHeight < 1.2*heightOffsetRatio):
                        #print "It passed the second X test"
                        betterBoundingBoxes = betterBoundingBoxes + [box]
                    else:
                        pass #print "It did not pass the second X test, width = ", width, "and secondWidth = ", secondWidth
                else:
                     pass#print "It did not pass the first X test x was: ", x, "and it had to be between ", secondX - 25, "and", secondX + 25
            else:
                pass#print "It did not pass the first Y test secondY - y was: ", secondY - y, "and the y difference was: ", yDifference
                       
    return betterBoundingBoxes

def filterByOtherTargetLift(goodBoundingBoxes, ratio, yOffsetRatio, heightOffsetRatio):
    betterBoundingBoxes = []
    if len(goodBoundingBoxes) < 2:
        return goodBoundingBoxes
    for box in goodBoundingBoxes:

        #print 'box: ',box
        if len(box) == 0:
            #print 'uh oh 1'
            continue
        x,y,width,height = box
        
        for secondBox in goodBoundingBoxes:
            
            if box == secondBox:
                
                continue
            if len(secondBox) == 0:

                #print 'uh oh 3'
                continue
            
            #print 'secondBox: ',secondBox
            #print 'len(secondBox): ', len(secondBox)
            #print 'len(goodBoundingBoxes): ',len(goodBoundingBoxes)
            secondX,secondY,secondWidth,secondHeight = secondBox

            #print 'width', width
            xDifference = width*ratio #Constant of proportionality of width of the 
            #retro Reflective to the width between the retro targets top left to top left
            #print 'xDifference is:', xDifference
            #print 'comparing: ', box, 'and', secondBox
            
            if xDifference - width*2 < secondX - x < xDifference:
                #print "passed X test"
                
                if secondY - yOffsetRatio*secondHeight < y < secondY + yOffsetRatio*secondHeight :
                    #print "passed Y test"
                    if (secondHeight-heightOffsetRatio*secondHeight < height < secondHeight + heightOffsetRatio*secondHeight or
                        height-heightOffsetRatio*height < secondHeight < height + heightOffsetRatio*height):
                        #print "passed Height test"
                        betterBoundingBoxes = betterBoundingBoxes + [box]
                        betterBoundingBoxes = betterBoundingBoxes + [secondBox]
                        
    return betterBoundingBoxes

def conjoinAnyBlobs(otherBoundingBoxesList,ratio):
    betterBoundingBoxes = []
    for box in otherBoundingBoxesList:
        ret, betterBoundingBox = checkForConjoiningBlobs(box,otherBoundingBoxesList,ratio)
        betterBoundingBoxes = betterBoundingBoxes + [betterBoundingBox]
        
    return betterBoundingBoxes

def checkForConjoiningBlobs(goodBoundingBox, otherBoundingBoxesList, ratio):
    betterBoundingBox = []
    x,y,width,height = goodBoundingBox

    #print 'the Length is: ', len(otherBoundingBoxesList)
    ret = False
    for box in otherBoundingBoxesList:
        secondX,secondY,secondWidth,secondHeight = box
        if box == goodBoundingBox:
            continue

         
        if ((x - width*ratio < secondX < x + width*ratio or x + width - width*ratio < secondX + secondWidth < x + width + width*ratio)):

            #print "Conjoining blobs: Passed X test"
         
            if y - 1.5*height < secondY < y:
                #print 'Conjoining blobs: Passed Y test'
                betterBoundingBox = (x,secondY,width,(y + height) - secondY)
                
                if ret:
                    #print "Error: Conjoined more than one blob"
                    return False, betterBoundingBox
                ret = True
        
    return ret, betterBoundingBox


#This is a tuning function

def drawBoundingBoxes (image, goodBoundingBoxes):
    copy = image.copy()
    for box in goodBoundingBoxes:
        x,y,width,height = box
        copy = cv2.rectangle(copy,(x,y),((x + width), (y + height)),(255,0,0), 3)
    #small = cv2.resize(copy, (0,0), fx = 0.2, fy = 0.2)
    
    #cv2.imshow("Processed Image", copy)
    

#These are the Math functions

def getRadiansToTurnFromOpticalAxis(boundingBoxOfTarget):
    x,y,width,height = boundingBoxOfTarget
    distanceFromCenterX = x - m_centerXOfImage
    radiansToTurn = math.atan(distanceFromCenterX/m_focalLengthOfCameraX)
    
    return radiansToTurn

def getRadiansToTurnHighGoalAndDistanceAwayShooter(boundingBoxOfTarget):
    x,y,width,height = boundingBoxOfTarget[0]
    betterBoundingBoxOfTarget = [x + width/2, y, width/2,height]
    radiansToTurnFromCamera = getRadiansToTurnFromOpticalAxis(betterBoundingBoxOfTarget)
    distanceAwayFromHighGoal = getDistanceAwayHighGoal(boundingBoxOfTarget)
    oppositeSide = math.sin(radiansToTurnFromCamera)*distanceAwayFromHighGoal
    adjacentSide = math.cos(radiansToTurnFromCamera)*distanceAwayFromHighGoal
    centerOfRobotAdjacent = adjacentSide + m_forwardOffsetOfCamera
    centerOfRobotOppositeSide = oppositeSide + m_lateralRightOffsetOfCamera
    centerOfRobotHypotenuse = math.sqrt(centerOfRobotAdjacent*centerOfRobotAdjacent + centerOfRobotOppositeSide*centerOfRobotOppositeSide)
    angleToTurnFromCenterOfRobot = math.atan(centerOfRobotOppositeSide/centerOfRobotAdjacent)
    deltaAngleFromShooter = math.atan(m_lateralRightOffsetOfShooter/centerOfRobotHypotenuse)
    angleToTurnFromShooter = angleToTurnFromCenterOfRobot - deltaAngleFromShooter
    parallelDistanceAway = math.sqrt(centerOfRobotHypotenuse*centerOfRobotHypotenuse - m_lateralRightOffsetOfShooter*m_lateralRightOffsetOfShooter)
    shooterDistanceAway = parallelDistanceAway - m_forwardOffsetOfShooter
    return angleToTurnFromShooter, shooterDistanceAway

def getDistanceAwayHighGoal(boundingBoxOfTarget):
    x,y,width,height = boundingBoxOfTarget[0]
    distanceFromCenterY = m_centerYOfImage - y
    elevationAngle = math.atan((distanceFromCenterY)/(m_focalLengthOfCameraY))
    offsetAddedElevationAngle = elevationAngle + m_radiansAngleofCamera
    distanceAwayHighGoalFromCamera = m_heightOfHighGoalTargetFromCamera/math.tan(offsetAddedElevationAngle) #Finding Adjacent; open to change
    return distanceAwayHighGoalFromCamera

def getDistanceAwayLift(boundingBoxOfTarget):

    #print "Bounding Box: ", boundingBoxOfTarget
    x,y,width,height = boundingBoxOfTarget
    distanceFromCenterY = m_centerYOfImage - y
    #print "m_centerYOfImage: ", m_centerYOfImage
    #print 'distanceFromCenterY', distanceFromCenterY
    #print 'm_focalLengthOfCameraY', m_focalLengthOfCameraY
    elevationAngle = math.atan((distanceFromCenterY)/(m_focalLengthOfCameraY))
    #print "elevationAngle", elevationAngle
    offsetAddedElevationAngle = elevationAngle + m_radiansAngleofCamera
    #print 'offsetAddedElevationAngle', offsetAddedElevationAngle*(180/math.pi)
    #print offsetAddedElevationAngle*180/math.pi
    #print math.tan(offsetAddedElevationAngle)
    #print
    distanceAwayLift = m_heightOfLiftTargetFromCamera/math.tan(offsetAddedElevationAngle) #Finding Adjacent; open to change
    #print 'distanceAwayLift', distanceAwayLift
    betterDistanceAwayLift = distanceAwayLift/math.cos(m_radiansAngleofCamera)
    return betterDistanceAwayLift

def get0(vector):
    return vector[0]

#Found on stack overflow; question 7446126
def getIntersectingPoint(line1, line2):
    origin1 = line1[2:4, :] #np.mat([line1[2], line1[3]])
    origin2 = line2[2:4, :] #np.mat([line2[2], line2[3]])
    d1 = line1[0:2, :] #np.mat([line1[0], line1[1]])
    d2 = line2[0:2, :] #np.mat([line2[0], line2[1]])
    x = origin2 - origin1
    #d1 = point1 - origin1
    #d2 = point2 - origin2
    cross = d1[0,0]*d2[1,0] - d1[1,0]*d2[0,0]   
    t1 = (x[0,0]*d2[1,0] - x[1,0]*d2[0,0])/ cross
    return origin1 + d1 * t1

def getBetterCoordinateMatrix(matrix):
    x = matrix[0][0]
    y = matrix[1][0]
    return [x,y]

def getRadiansToTurnLiftAndDistanceToDriveForwardAndLaterally(picture, boundingBoxesOfTargets):
    imgpoints = []
    if len(boundingBoxesOfTargets) == 1:
        correctTarget = boundingBoxesOfTargets[0]
    else:
        firstBoundingBox = boundingBoxesOfTargets[0]
        secondBoundingBox = boundingBoxesOfTargets[1]

        firstX, firstY, firstWidth, firstHeight = firstBoundingBox
        secondX, secondY, secondWidth, secondHeight = secondBoundingBox
        #print "firstBoundingBox", firstBoundingBox
        #print "secondBoundingBox", secondBoundingBox
        #print "(firstX + secondX + secondWidth)/2", (firstX + secondX + secondWidth)/2
        #print "m_centerXOfImage", m_centerXOfImage
        if (firstX + secondX + secondWidth)/2 < m_centerXOfImage:
            if firstX > secondX:
                correctTarget = firstBoundingBox
                incorrectTarget = secondBoundingBox

            else:
                correctTarget = secondBoundingBox
                incorrectTarget = firstBoundingBox
 
        
        else:
            if firstX > secondX:
                correctTarget = secondBoundingBox
                incorrectTarget = firstBoundingBox

            else:
                correctTarget = firstBoundingBox
                incorrectTarget = secondBoundingBox

        if correctTarget[0] > incorrectTarget[0]:
            leftTarget = False

        else:
            leftTarget = True


        if leftTarget:
            objPoints = np.matrix([[-5.125,0,15.75],[-3.125,0,10.75],[-5.125,0,10.75],[-3.125,0,15.75]])
        else:
            objPoints = np.matrix([[3.125,0,15.75],[5.125,0,10.75],[3.125,0,10.75],[5.125,0,15.75]])

        x,y,width,height = correctTarget
        offset = height*0.21212121
        tempImage = picture[y - offset:y+height+offset, x-offset:x+width+offset]
                
        correctColorImage = filterColors(tempImage,50,240,10,65,255,80)
        correctColorImage = cv2.GaussianBlur(correctColorImage, (5,5),0)
        correctColorImage, contours, hierarchy = cv2.findContours(correctColorImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
        leftLinePoints = []
        rightLinePoints = []
        topLinePoints = []
        bottomLinePoints = []

        maxlength = -1
        for contour in contours:
            lengthOfContour = len(contour)
            if lengthOfContour > maxlength:
                maxlength = lengthOfContour
                maxLengthContour = contour
                
        for coordinate in maxLengthContour:
            coordinateX, coordinateY = coordinate[0]
                
            if offset/2 < coordinateX < offset + width*0.5 and offset + height*0.25 < coordinateY < height*0.75 + offset:
                leftLinePoints.append(coordinate)
                
            elif offset/2 < coordinateY < offset + height*0.5 and offset + width*0.25 < coordinateX < width*0.75 + offset:
                topLinePoints.append(coordinate)

            elif width*0.5 + offset< coordinateX < width + 1.5*offset and offset + height*0.25 < coordinateY < height*0.75 + offset:
                rightLinePoints.append(coordinate)

            elif height*0.5 + offset < coordinateY < height + 1.5*offset and offset + width*0.25 < coordinateX < width*0.75 + offset:
                bottomLinePoints.append(coordinate)

        leftLine = cv2.fitLine(np.array(leftLinePoints), cv2.DIST_L2, 0, 0,0)
        rightLine = cv2.fitLine(np.array(rightLinePoints), cv2.DIST_L2, 0, 0,0)
        topLine = cv2.fitLine(np.array(topLinePoints), cv2.DIST_L2, 0, 0,0)
        bottomLine = cv2.fitLine(np.array(bottomLinePoints), cv2.DIST_L2, 0, 0,0)
       
        topLeftCorner = getIntersectingPoint(leftLine, topLine)
        topRightCorner = getIntersectingPoint(topLine, rightLine)
        bottomRightCorner = getIntersectingPoint(bottomLine, rightLine)
        bottomLeftCorner = getIntersectingPoint(bottomLine, leftLine) 

            
        topLeftCorner = getBetterCoordinateMatrix(topLeftCorner)
        topRightCorner = getBetterCoordinateMatrix(topRightCorner)
        bottomRightCorner = getBetterCoordinateMatrix(bottomRightCorner)
        bottomLeftCorner = getBetterCoordinateMatrix(bottomLeftCorner)

            
        topLeftCorner = [topLeftCorner[0] + x - offset, topLeftCorner[1] + y - offset]
        topRightCorner = [topRightCorner[0] + x - offset, topRightCorner[1] + y - offset]
        bottomRightCorner = [bottomRightCorner[0] + x - offset, bottomRightCorner[1] + y - offset]
        bottomLeftCorner = [bottomLeftCorner[0] + x - offset, bottomLeftCorner[1] + y - offset]
            
        imgpoints.append(topLeftCorner)
        imgpoints.append(bottomRightCorner)
        imgpoints.append(bottomLeftCorner)
        imgpoints.append(topRightCorner)
        
        
    imgpoints = np.array(imgpoints)
    cv2.destroyAllWindows()
    ret, targetRvec, targetTvec = cv2.solvePnP(objPoints, imgpoints, m_cameraMatrix, m_distCoeffs, None, None, False, cv2.SOLVEPNP_ITERATIVE)
    
    simpleVec = np.append(targetTvec[0],targetTvec[1])#np.array(targetTvec)#map(get0, targetTvec)
    simpleVec = np.append(simpleVec, targetTvec[2])
    targetR,s = cv2.Rodrigues(targetRvec)
    robotTvec = m_RCamera.dot(simpleVec) + m_tvecCamera.T
    robotR = targetR*m_RCamera
    robotTvec = robotTvec[0]
    
    robotTvecAfterTurning = robotR.dot(robotTvec)
    return robotR, robotTvecAfterTurning

def getDistanceToMoveLaterallyAndDistanceToMoveForwardBoundingBox(boundingBoxOfTarget):
    oppositeAngle = getRadiansToTurnFromOpticalAxis(boundingBoxOfTarget)
    #print "getRadiansToTurnFromOpticalAxis", oppositeAngle
    distanceAwayLift = getDistanceAwayLift(boundingBoxOfTarget)
    #print "distanceAwayLift", distanceAwayLift
    #print distanceAwayLift
    distanceToMoveLaterally = math.sin(oppositeAngle)*distanceAwayLift
    distanceToMoveForwardLift = math.cos(oppositeAngle)*distanceAwayLift
    return distanceToMoveLaterally, distanceToMoveForwardLift

def getDistanceToMoveLaterallyAndDistanceToMoveForwardLift(boundingBoxesOfTargets):
    #print 'len(boundingBoxesOfTargets)', len(boundingBoxesOfTargets)
    #print 'boundingBoxesOfTargets', boundingBoxesOfTargets
    if len(boundingBoxesOfTargets) == 1:
        
        boundingBoxOfTarget = boundingBoxesOfTargets[0]
        distanceToMoveLaterally, distanceToMoveForward = getDistanceToMoveLaterallyAndDistanceToMoveForwardBoundingBox(boundingBoxOfTarget)

        #print "initial ", distanceToMoveLaterally

        if distanceToMoveLaterally < 0:
            print 'leftTarget False'
            distanceToMoveLaterally = distanceToMoveLaterally - 3.135
        else:
            print 'leftTarget True'
            distanceToMoveLaterally = distanceToMoveLaterally + 5.135
        distanceToMoveLaterally = distanceToMoveLaterally + m_rightOffsetOfGearPlacerFromCamera
        distanceToMoveForward = distanceToMoveForward + m_forwardOffsetOfGearPlacerFromCamera
        #distanceToMoveLaterally = distanceToMoveLaterally-2.5
        
        return distanceToMoveLaterally, distanceToMoveForward
    
    firstBoundingBox = boundingBoxesOfTargets[0]
    secondBoundingBox = boundingBoxesOfTargets[1]

    firstX, firstY, firstWidth, firstHeight = firstBoundingBox
    secondX, secondY, secondWidth, secondHeight = secondBoundingBox
    #print "firstBoundingBox", firstBoundingBox
    #print "secondBoundingBox", secondBoundingBox
    #print "(firstX + secondX + secondWidth)/2", (firstX + secondX + secondWidth)/2
    #print "m_centerXOfImage", m_centerXOfImage
    if (firstX + secondX + secondWidth)/2 < m_centerXOfImage:
        if firstX > secondX:
            correctTarget = firstBoundingBox
            incorrectTarget = secondBoundingBox

        else:
            correctTarget = secondBoundingBox
            incorrectTarget = firstBoundingBox
 
        
    else:
        if firstX > secondX:
            correctTarget = secondBoundingBox
            incorrectTarget = firstBoundingBox

        else:
            correctTarget = firstBoundingBox
            incorrectTarget = secondBoundingBox

    if correctTarget[0] > incorrectTarget[0]:
        leftTarget = False

    else:
        leftTarget = True
        
    distanceToMoveLaterally,distanceToMoveForwardLift = getDistanceToMoveLaterallyAndDistanceToMoveForwardBoundingBox(correctTarget)
    print 'leftTarget', leftTarget
    if leftTarget:
        distanceToMoveLaterally = distanceToMoveLaterally + 5.125
    else:
        distanceToMoveLaterally = distanceToMoveLaterally - 3.125

    #distanceToMoveLaterally = distanceToMoveLaterally-2.5
    return distanceToMoveLaterally, distanceToMoveForwardLift

def initNetworkTables():
    logging.basicConfig(level=logging.DEBUG)

    ip = "roborio-4905-frc.local" #"10.49.5.77"
    NetworkTables.initialize(server=ip)
    #NetworkTables.setIPAddress("192.168.7.71")
    cameraNT = NetworkTables.getTable("VisionProcessing")
    return cameraNT
    
def putDataOnNetworkTablesLift(networkTable, booleanFoundTarget, timestampLift,robotTimestampLift,radiansToTurnLift,distanceToMoveLaterallyLift,distanceToDriveForwardLift):
    networkTable.putBoolean('foundLiftTarget', booleanFoundTarget)
    networkTable.putNumber('radiansToTurnLift', radiansToTurnLift)
    networkTable.putNumber('distanceToDriveLaterallyLift', distanceToMoveLaterallyLift)
    networkTable.putNumber('distanceToDriveForwardLift', distanceToDriveForwardLift)
    networkTable.putNumber('timestampLift', timestampLift)
    networkTable.putNumber("robotTimestampLift", robotTimestampLift)
    
def putDataOnNetworkTablesHighGoal(networkTable, booleanFoundTarget, timestampHighGoal,robotTimestampHighGoal,radiansToTurnHighGoal,distanceAwayHighGoal):
    networkTable.putBoolean('foundHighGoalTarget', booleanFoundTarget)
    networkTable.putNumber('radiansToTurnHighGoal', radiansToTurnHighGoal)
    networkTable.putNumber('distanceAwayHighGoal', distanceAwayHighGoal)
    networkTable.putNumber('timestampHighGoal', timestampHighGoal)
    networkTable.putNumber('robotTimestampHighGoal', robotTimestampHighGoal)

def getDataFromNetworktables(networkTable):
    

    turnOnRet = networkTable.getBoolean("RobotEnabled", False)
    
    timestampRet = networkTable.getBoolean("TimestampRet", False)
    #print "timestampRet: ", timestampRet
    timestamp = networkTable.getNumber('Timestamp', 0.0)
    return turnOnRet, timestampRet, timestamp

def getRobotParallelStatus(networkTable):
    return networkTable.getBoolean("ParallelStatus", True)#This is true because this networktable is not created on the java side of some branches so we need this to be true so that those branches of code still work

def setShortTermMemory(newTimestamp, image):
    m_shortTermMemory.append((newTimestamp, image))
    while newTimestamp - m_shortTermMemory[0][0] > m_microsecondsToSaveMemory:
        m_shortTermMemory.popleft()
    
def saveImage(timestamp, networkTable):
    while timestamp - m_shortTermMemory[0][0] > 0:
        m_shortTermMemory.popleft()
    
    if m_shortTermMemory[0][0] == timestamp:
        #print "timestamp", timestamp
        cv2.imwrite("/home/pi/test/AndromedaVision/FailedImageProcessingImages/Image%d.png" % timestamp, m_shortTermMemory[0][1])
        m_shortTermMemory.popleft()

        #print "Save Image: ", timestamp
        
    networkTable.putBoolean('TimestampRet', False)
    
def dispatchCommands(timestamp, cameraStream, networkTable):
    setShortTermMemory(timestamp, cameraStream)
    turnOnRet, timestampRet, timestamp = getDataFromNetworktables(networkTable)
    #print "timestampRet", timestampRet
    #print "timestamp", timestamp
    #print "not turnOnRet", not turnOnRet
    if not turnOnRet:
        m_libc.sync()

    if timestampRet:
        saveImage(timestamp, networkTable)
   
def main():
    initializedCameraStream = cameraStreamInit()

    #print "2"
    sd = initNetworkTables()
    if m_typeOfCamera == 'Shooter':
        while True:
            
            timestamp,cameraStream, timestampForPi = getCameraStream(initializedCameraStream, sd)
            setShortTermMemory(timestamp, cameraStream)

            #print 'timestamp', timestamp
            retHighGoal,highGoalTarget = findHighGoalTarget(cameraStream)
            if retHighGoal:
                radiansToTurnHighGoalFromShooter, distanceAwayHighGoalFromShooter = getRadiansToTurnHighGoalAndDistanceAwayShooter(highGoalTarget)
                putDataOnNetworkTablesHighGoal(sd,True,timestampForPi,radiansToTurnHighGoalFromShooter,distanceAwayHighGoalFromShooter)
            else:
                putDataOnNetworkTablesHighGoal(sd,False,timestampForPi,0,0)
            dispatchCommands(timestampForPi, cameraStream, sd)
            
    else:
        while True:

            initialrobotParallelStatus = getRobotParallelStatus(sd)
            timestamp,cameraStream, timestampForPi = getCameraStream(initializedCameraStream, sd)
            afterPicRobotParallelStatus = getRobotParallelStatus(sd)
            setShortTermMemory(timestampForPi, cameraStream)
            #print 'timestamp', timestamp
            retLift,liftTargets = findLiftTarget(cameraStream)
            print 'retLift', retLift
            if retLift and initialrobotParallelStatus and afterPicRobotParallelStatus:
                #robotR, robotTvecAfterTurning= getRadiansToTurnLiftAndDistanceToDriveForwardAndLaterally(cameraStream, cameraStream, liftTargets)

                #print 'robotTvecAfterTurning', robotTvecAfterTurning
                #eulerAngles = rotationMatrixToEulerAngles(robotR)
                #radiansToTurnLift = eulerAngles[2]
                #distanceToMoveLaterallyLift = robotTvecAfterTurning[0]
                #distanceToDriveForwardLift = robotTvecAfterTurning[1]
                distanceToMoveLaterallyLift, distanceToDriveForwardLift = getDistanceToMoveLaterallyAndDistanceToMoveForwardLift(liftTargets)
                radiansToTurnLift = 0
                putDataOnNetworkTablesLift(sd,True,timestampForPi,timestamp,radiansToTurnLift,distanceToMoveLaterallyLift,distanceToDriveForwardLift)
                #degreesToTurn = radiansToTurnLift*(180/math.pi)

                #print "degreesToTurnLift: ", degreesToTurn
                print 'distanceToMoveLaterallyLift', distanceToMoveLaterallyLift, " Inches"
                print 'distanceToDriveForwardLift', distanceToDriveForwardLift, " Inches"
           
            else:

                putDataOnNetworkTablesLift(sd,False,timestampForPi,timestamp,0,0,0)
                #print "Working"
            dispatchCommands(timestampForPi, cameraStream, sd)    
            
main()


#pics = '/home/pi/Pictures/PicsFromUNH'
#for filename in os.listdir(pics):
 #   fullFileName = os.path.join(pics, filename)
  #  print 'fullFileName', fullFileName
   # pic = cv2.imread(fullFileName)
    #retLift, liftTargets = findLiftTarget(pic)
    #distanceToMoveLaterallyLift, distanceToDriveForwardLift = getDistanceToMoveLaterallyAndDistanceToMoveForwardLift(liftTargets, pic)
    #print "distanceToMoveLaterallyLift", distanceToMoveLaterallyLift
    #cv2.waitKey(0)
    
#cv2.destroyAllWindows()
#pic = cv2.imread("/home/pi/Pictures/BadImages/Image937486405.png")
#retLift, liftTargets = findLiftTarget(pic)
#distanceToMoveLaterallyLift, distanceToDriveForwardLift = getDistanceToMoveLaterallyAndDistanceToMoveForwardLift(liftTargets, pic)
#print "distanceToMoveLaterallyLift", distanceToMoveLaterallyLift
#cv2.waitKey(0)
    
cv2.destroyAllWindows()
