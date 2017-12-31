import cv2
import numpy as np
import datetime
import FiltrationType

m_minH = 58
m_minS = 88
m_minV = 5
m_maxH = 65
m_maxS = 255
m_maxV = 186

m_numberOfContours = 4

m_minHeightSize = 12
m_maxHeightSize = 2000
m_minWidthSize = 24
m_maxWidthSize = 2000

m_lowLength2WidthRatio = 1.5
m_highLength2WidthRatio = 5.5

m_black2WhiteRatioMin = -1
m_black2WhiteRatioMax = 2.5
def findTarget(img, testMode = False, filtrationType = None):
    #Runs all the filtiration methods
    
    correctColorImage = filterColors(img)
    
    if testMode and (filtrationType is None or filtrationType == FiltrationType.colorFiltered):
        filtrationType = None
        channel3CorrectColorImage = cv2.cvtColor(correctColorImage, cv2.COLOR_GRAY2RGB)
        copy = cv2.addWeighted(img, 0.5, channel3CorrectColorImage, 0.5,1.0)
        cv2.imshow("Color Filtered", copy)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False, [],FiltrationType.colorFiltered
        
    preparedImage = prepareImage(correctColorImage)
    
    copy = preparedImage.copy() #need to do this because the findContours function alters the source image

    correctNumberOfContoursList = filterContours(copy)
    if testMode and (filtrationType is None or filtrationType == FiltrationType.contourFiltered):
        filtrationType = None
        countourFilteredImage = drawBoundingBoxes(img, correctNumberOfContoursList)
        cv2.imshow("Contour Filtered", countourFilteredImage)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False, correctNumberOfContoursList,FiltrationType.contourFiltered
        
    correctSizeList = filterSize(correctNumberOfContoursList)
    if testMode and (filtrationType is None or filtrationType == FiltrationType.sizeFiltered):
        filtrationType = None
        sizeFilteredImage = drawBoundingBoxes(img, correctSizeList)
        cv2.imshow("Size Filtered", sizeFilteredImage)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False, correctSizeList,FiltrationType.sizeFiltered
        
    correctBlack2WhiteRatioList = filterBlack2WhiteRatio(correctSizeList,preparedImage) 
    if testMode and (filtrationType is None or filtrationType == FiltrationType.black2WhiteFiltered):
        filtrationType = None
        black2WhiteFilteredImage = drawBoundingBoxes(img, correctBlack2WhiteRatioList)
        cv2.imshow("Black2White Filtered", black2WhiteFilteredImage)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False, correctBlack2WhiteRatioList,FiltrationType.black2WhiteFiltered
        
    correctLengthToWidthRatioList = filterLength2WidthRatio(correctBlack2WhiteRatioList)
    if testMode and (filtrationType is None or filtrationType == FiltrationType.length2WidthFiltered):
        filtrationType = None
        length2WidthFilteredImage = drawBoundingBoxes(img, correctLengthToWidthRatioList)
        cv2.imshow("Length2Width Filtered", length2WidthFilteredImage)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False, correctLengthToWidthRatioList,FiltrationType.length2WidthFiltered
        
    return len(correctLengthToWidthRatioList) == 1, correctLengthToWidthRatioList, None
    
def prepareImage(image):
    #Cancels out very small bits of noice by blurring the image and then eroding it
    
    gaussianBlurImage = cv2.GaussianBlur(image,(3,3),1.6)

    return gaussianBlurImage


def filterColors(image, minHParam=None, minSParam=None, minVParam=None, maxHParam=None, maxSParam=None, maxVParam=None):

    minH = m_minH if minHParam is None else minHParam
    minS = m_minS if minSParam is None else minSParam
    minV = m_minV if minVParam is None else minVParam
    maxH = m_maxH if maxHParam is None else maxHParam
    maxS = m_maxS if maxSParam is None else maxSParam
    maxV = m_maxS if maxVParam is None else maxVParam
    #Filters out all colors but green; Returns color filtered image
    HSVImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSVImg,(minH,minS,minV),(maxH,maxS,maxV))

    return mask

def filterContours(image, numberOfContoursParam = None):
    numberOfContours = m_numberOfContours if numberOfContoursParam is None else numberOfContoursParam
    #Filters out all "Blobs" with less than "numberOfContours" contours 
    #Returns BOUNDING BOXES of "Blobs" having over 8 contours
    img3,contours,hierarchy = cv2.findContours(image, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    goodBoundingBoxes = []
    for box in contours:
        if len(box)>= numberOfContours:
            goodBoundingBoxes = goodBoundingBoxes + [cv2.boundingRect(box)]
    return goodBoundingBoxes
    #Returns BOUNDING BOXES!!!!

def filterSize(goodBoundingBoxes, minHeightSizeParam = None, maxHeightSizeParam = None, minWidthSizeParam = None, maxWidthSizeParam = None):
    minHeightSize = m_minHeightSize if minHeightSizeParam is None else minHeightSizeParam
    maxHeightSize = m_maxHeightSize if maxHeightSizeParam is None else maxHeightSizeParam
    minWidthSize = m_minWidthSize if minWidthSizeParam is None else minWidthSizeParam
    maxWidthSize = m_maxWidthSize if maxWidthSizeParam is None else maxWidthSizeParam
    #Filters out "Blobs" that are way too big or way too small
    betterBoundingBoxes = []
    for box in goodBoundingBoxes:
        width =  box[2]
        height =  box[3]
        if minHeightSize < height < maxHeightSize and minWidthSize < width < maxWidthSize:
            betterBoundingBoxes = betterBoundingBoxes + [box]
    return betterBoundingBoxes

def filterLength2WidthRatio(goodBoundingBoxes, lowLength2WidthRatioParam = None, highLength2WidthRatioParam = None):
    lowLength2WidthRatio = m_lowLength2WidthRatio if lowLength2WidthRatioParam is None else lowLength2WidthRatioParam
    highLength2WidthRatio = m_highLength2WidthRatio if highLength2WidthRatioParam is None else highLength2WidthRatioParam
    #Filters out all "Blobs" with length to width ratios not between lowLengthToWidthRatio and highLengthToWidthRatio
    betterBoundingBoxes = []
    for box in goodBoundingBoxes:
        width =  box[2]
        height =  box[3]
        if lowLength2WidthRatio < (width + 0.0)/ (height+ 0.0) < highLength2WidthRatio:
            betterBoundingBoxes = betterBoundingBoxes +  [box]
    return betterBoundingBoxes

def filterBlack2WhiteRatio(goodBoundingBoxes, image, black2WhiteRatioMinParam = None, black2WhiteRatioMaxParam = None):
    black2WhiteRatioMin = m_black2WhiteRatioMin if black2WhiteRatioMinParam is None else black2WhiteRatioMinParam
    black2WhiteRatioMax = m_black2WhiteRatioMax if black2WhiteRatioMaxParam is None else black2WhiteRatioMaxParam
    #Filters out all "Blobs" that do not have a ratio of white to black pixels between blackToWhiteRatioMin - blackToWhiteRatioMax 
    betterBoundingBoxes = []
    for box in goodBoundingBoxes:
        x,y,width,height = box
        tempImage = image[y+height/2:y+height, x:x+width]
        
        numberOfWhitePixels = cv2.countNonZero(tempImage)

        if black2WhiteRatioMin < ((width*(height/2) - numberOfWhitePixels+ 0.0))/(numberOfWhitePixels) < black2WhiteRatioMax:#number of black pixels for every white pixel
            betterBoundingBoxes = betterBoundingBoxes + [box]
        
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

    ret = False
    for box in otherBoundingBoxesList:
        secondX,secondY,secondWidth,secondHeight = box
        if box == goodBoundingBox:
            continue

         
        if ((x - width*ratio < secondX < x + width*ratio or x + width - width*ratio < secondX + secondWidth < x + width + width*ratio)):

            
            if y - 1.5*height < secondY < y:
                betterBoundingBox = (x,secondY,width,(y + height) - secondY)
                
                if ret:
                    return False, betterBoundingBox
                ret = True
        
    return ret, betterBoundingBox


#This is a tuning function

def drawBoundingBoxes (image, goodBoundingBoxes):
    copy = image.copy()
    for box in goodBoundingBoxes:
        x,y,width,height = box
        copy = cv2.rectangle(copy,(x,y),((x + width), (y + height)),(255,0,0), 3)

    return copy

def setGlobalVariables(filtrationType, listOfParameterValues):
    global m_minH
    global m_minS
    global m_minV
    global m_maxH
    global m_maxS
    global m_maxV

    global m_numberOfContours

    global m_minHeightSize
    global m_maxHeightSize
    global m_minWidthSize
    global m_maxWidthSize

    global m_lowLength2WidthRatio
    global m_highLength2WidthRatio

    global m_black2WhiteRatioMin
    global m_black2WhiteRatioMax

    if filtrationType == FiltrationType.colorFiltered:
        if listOfParameterValues[0] < m_minH:
            m_minH = listOfParameterValues[0]
            
        if listOfParameterValues[1] < m_minS:
            m_minS = listOfParameterValues[1]
        if listOfParameterValues[2] < m_minV:
            m_minV = listOfParameterValues[2]

        if listOfParameterValues[3] > m_maxH:
            m_maxH = listOfParameterValues[3]
        if listOfParameterValues[4] > m_maxS:
            m_maxS = listOfParameterValues[4]
        if listOfParameterValues[5] > m_maxV:
            m_maxV = listOfParameterValues[5]
        
    elif filtrationType == FiltrationType.contourFiltered:
        if listOfParameterValues[0] < m_numberOfContours:
            m_numberOfContours = listOfParameterValues[0]
        
    elif filtrationType == FiltrationType.sizeFiltered:
        if listOfParameterValues[0] < m_minHeightSize:
            m_minHeightSize = listOfParameterValues[0]
        if listOfParameterValues[1] > m_maxHeightSize:
            m_maxHeightSize = listOfParameterValues[1]
        if listOfParameterValues[2] < m_minWidthSize:
            m_minWidthSize = listOfParameterValues[2]
        if listOfParameterValues[3] > m_maxWidthSize:
            m_maxWidthSize = listOfParameterValues[3]
        
    elif filtrationType == FiltrationType.black2WhiteFiltered:
        if listOfParameterValues[0]/10.0 < m_black2WhiteRatioMin:
            m_black2WhiteRatioMin = listOfParameterValues[0]/10.0
        if listOfParameterValues[1]/10.0 > m_black2WhiteRatioMax:
            
            m_black2WhiteRatioMax = listOfParameterValues[1]/10.0
            

    elif filtrationType == FiltrationType.length2WidthFiltered:
        if listOfParameterValues[0]/10.0 < m_lowLength2WidthRatio:
            m_lowLength2WidthRatio = listOfParameterValues[0]/10.0
        if listOfParameterValues[1]/10.0 > m_highLength2WidthRatio:
            m_highLength2WidthRatio = listOfParameterValues[1]/10.0

def printGlobalVariables():
    global m_minH
    global m_minS
    global m_minV
    global m_maxH
    global m_maxS
    global m_maxV

    global m_numberOfContours

    global m_minHeightSize
    global m_maxHeightSize
    global m_minWidthSize
    global m_maxWidthSize

    global m_lowLength2WidthRatio
    global m_highLength2WidthRatio

    global m_black2WhiteRatioMin
    global m_black2WhiteRatioMax

    print 'm_minH: ' , m_minH
    print 'm_minS: ' , m_minS
    print 'm_minV: ' , m_minV
    print 'm_maxH: ' , m_maxH
    print 'm_maxS: ' , m_maxS
    print 'm_maxV: ' , m_maxV

    print 'm_numberOfContours: ' , m_numberOfContours

    print 'm_minHeightSize: ' , m_minHeightSize
    print 'm_maxHeightSize: ' , m_maxHeightSize
    print 'm_minWidthSize: ' , m_minWidthSize
    print 'm_maxWidthSize: ' , m_maxWidthSize

    print 'm_lowLength2WidthRatio: ' , m_lowLength2WidthRatio
    print 'm_highLength2WidthRatio: ' , m_highLength2WidthRatio

    print 'm_black2WhiteRatioMin: ' , m_black2WhiteRatioMin
    print 'm_black2WhiteRatioMax: ' , m_black2WhiteRatioMax
    
