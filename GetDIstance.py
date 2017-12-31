#These are the Math functions
class Distance:
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
                #print 'leftTarget False'
                distanceToMoveLaterally = distanceToMoveLaterally - 3.135
            else:
                #print 'leftTarget True'
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
        #print 'leftTarget', leftTarget
        if leftTarget:
            distanceToMoveLaterally = distanceToMoveLaterally + 5.125
        else:
            distanceToMoveLaterally = distanceToMoveLaterally - 3.125

        #distanceToMoveLaterally = distanceToMoveLaterally-2.5
        return distanceToMoveLaterally, distanceToMoveForwardLift

