import Filtration
import cv2
import os

class UnitCases:
    
    def listCases(self,unitCaseImageDirectory):
 
        for imageName in os.listdir(unitCaseImageDirectory):
            if imageName[-4:] == ".png":
                imageFullFileName = os.path.join(unitCaseImageDirectory, imageName)
                foundTargetFullFileName = os.path.join(unitCaseImageDirectory, imageName[:-4] + ".txt")

                with open(foundTargetFullFileName) as f:
                    foundTargetString = f.read()
                    
                foundTarget = foundTargetString == "T" or foundTargetString == "t" or foundTargetString == " T" or foundTargetString ==  " t"
                with open(imageFullFileName) as f:
                    image = cv2.imread(imageFullFileName)
                 
                yield foundTarget,image
            
    
    def __init__(self, unitCaseImageDirectory):
        totalImages = 0
        correctImages = 0
        wrongImages = 0
        for unitCaseBooleanFoundTarget, unitCaseImage in self.listCases(unitCaseImageDirectory):
            foundTarget, correctTargetList = Filtration.findTarget(unitCaseImage)
            totalImages += 1
            if foundTarget == unitCaseBooleanFoundTarget:
                correctImages+=1

            else:
                wrongImages+=1
                cv2.imwrite("/Users/seandoyle/git/AndromedaVision/FailedImageProcessingImages12-28/image%d.png" % wrongImages, unitCaseImage)
                f = open('/Users/seandoyle/git/AndromedaVision/FailedImageProcessingImages12-28/image%d.txt' % wrongImages, "w")
                if unitCaseBooleanFoundTarget:
                    f.write("T")
                else:
                    f.write("F")
                f.close

        print "The number correct is: " , correctImages
        print "The total images is: ", totalImages

path = '/Users/seandoyle/git/AndromedaVision/ImageUnitCases12-28'
unitCases = UnitCases(path)

