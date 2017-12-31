import Filtration
import cv2
import os
import FiltrationType

def debugFailedUnitCases():
    path = '/Users/seandoyle/git/AndromedaVision/FailedImageProcessingImages12-28/'
    for imageName in os.listdir(path):
        imageFullFileName = os.path.join(path, imageName)
        if imageFullFileName[-4:] != '.png':
            continue
        image = cv2.imread(imageFullFileName)
        filtrationType = FiltrationType.colorFiltered
        while filtrationType is not None or filtrationType <= 4:
            computerFoundTarget, listOfTargets, filtrationType = Filtration.findTarget(image,True,filtrationType)
            if filtrationType is None:
                break
            listOfTuningParameters = []
            
            min2MaxRange = 0

            if filtrationType == FiltrationType.colorFiltered:
                listOfTuningParameters = ['minH', 'minS', 'minV','maxH', 'maxS', 'maxV']
                min2MaxRange = 255
            
            elif filtrationType == FiltrationType.contourFiltered:
                listOfTuningParameters = ['minCountours']
                min2MaxRange = 10
                
            elif filtrationType == FiltrationType.sizeFiltered:
                listOfTuningParameters = ['minHeightSize','maxHeightSize', 'minWidthSize', 'maxWidthSize']
                min2MaxRange = 500

            elif filtrationType == FiltrationType.black2WhiteFiltered:
                listOfTuningParameters = ['minBlack2WhiteRatio','maxBlack2WhiteRatio']
                min2MaxRange = 200
                

            elif filtrationType == FiltrationType.length2WidthFiltered:
                listOfTuningParameters = ['minLength2WidthRatio','maxLength2WidthRatio']
                min2MaxRange = 100

            setupImageWindow(listOfTuningParameters, min2MaxRange)
            
            tuneImageLive(image, filtrationType,listOfTuningParameters)

            filtrationType = filtrationType + 1

            
    cv2.destroyAllWindows()
    
    Filtration.printGlobalVariables()
    

def setupImageWindow(listOfTuningParameters, min2MaxRange):
    cv2.namedWindow("Processed Image")
    for parameter in listOfTuningParameters:
        cv2.createTrackbar(parameter, 'Processed Image',0,min2MaxRange,null)

def null(x):
    pass

def tuneImageLive(img, filtrationType, listOfTuningParameters):
    while True:
        
        listOfParameterValues = []
        for parameter in listOfTuningParameters:
            parameterValue = cv2.getTrackbarPos(parameter,'Processed Image')
            listOfParameterValues = listOfParameterValues + [parameterValue]

        if filtrationType == FiltrationType.colorFiltered:
            colorFilteredImage = Filtration.filterColors(img, listOfParameterValues[0],listOfParameterValues[1],listOfParameterValues[2],listOfParameterValues[3],listOfParameterValues[4],listOfParameterValues[5])

        elif filtrationType == FiltrationType.contourFiltered:
            preparedImage = Filtration.prepareImage(Filtration.filterColors(img))
            newList = Filtration.filterContours(preparedImage, listOfParameterValues[0])
            
        elif filtrationType == FiltrationType.sizeFiltered:
            preparedImage = Filtration.prepareImage(Filtration.filterColors(img))
            filterContoursList = Filtration.filterContours(preparedImage)
            newList = Filtration.filterSize(filterContoursList, listOfParameterValues[0], listOfParameterValues[1], listOfParameterValues[2], listOfParameterValues[3])
            
        elif filtrationType == FiltrationType.black2WhiteFiltered:
            preparedImage = Filtration.prepareImage(Filtration.filterColors(img))
            filterContoursList = Filtration.filterContours(preparedImage)
            filterSizeList = Filtration.filterSize(filterContoursList)
            newList = Filtration.filterBlack2WhiteRatio(filterSizeList, preparedImage, listOfParameterValues[0]/10.0, listOfParameterValues[1]/10.0)
            
            
        elif filtrationType == FiltrationType.length2WidthFiltered:
            preparedImage = Filtration.prepareImage(Filtration.filterColors(img))
            filterContoursList = Filtration.filterContours(preparedImage)
            filterSizeList = Filtration.filterSize(filterContoursList)
            filterBlack2WhiteList = Filtration.filterBlack2WhiteRatio(filterSizeList, preparedImage)
            newList = Filtration.filterLength2WidthRatio(filterBlack2WhiteList, listOfParameterValues[0]/10.0, listOfParameterValues[1]/10.0)

        if filtrationType != FiltrationType.colorFiltered:
            copy = Filtration.drawBoundingBoxes(img, newList)
        else:
            colorFilteredImage = cv2.cvtColor(colorFilteredImage, cv2.COLOR_GRAY2RGB)
            copy = cv2.addWeighted(img, 0.5, colorFilteredImage, 0.5,1.0)
            
        cv2.imshow("Processed Image", copy)
        key = cv2.waitKey(0)
        if key == ord('q'):
            Filtration.setGlobalVariables(filtrationType, listOfParameterValues)
            
            cv2.destroyAllWindows()
            break
        #Try again on any other key


debugFailedUnitCases()
cv2.destroyAllWindows()
