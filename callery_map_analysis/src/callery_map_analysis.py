#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import math
import requests
from matplotlib import pyplot as plt

#-Define Global Variables-------------------------------------------------------

resolution = 0.05
greenPx = [0, 255, 0]

# Building Analysis
dilationIterations = 3
erosionIterations = 2
kernelSize = 5
buildingArcLength = 0.001
badContPerimeter = 6

# Room Analysis
numCornToDetect = 100
roomArcLength = 0.05
roomAreaScale = 1.03
roomPerimScale = 1.015
# Quad doors
quadSet = set()
quadDistMin = 0.3 / resolution
quadDistMax = 1.8 / resolution
quadFillRatio = 0.9
# Pair doors
pairSet = set()
pairFillRatio = 0.95
pairGap = 4

#-Functions---------------------------------------------------------------------

# return pixel distance between two points
def getDist(p1, p2):
    dist = math.sqrt(((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2))
    return dist

# Communicate data over IFTTT protocol
def ifttt_details(first, second, third):
    report = {}
    report["value1"] = first
    report["value2"] = second
    report["value3"] = third
    requests.post("https://maker.ifttt.com/trigger/geom_details/with/key/nS8o2Md4dpGoAOtCKg-b2xVG-O3id3Iapi6cwJItolI", data=report)
    return

#-Building Analysis Functions---------------------------------------------------

# convert image to black and white
def convertImageBandW(image):
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh

# dilation and erosion removes gaps in map contours
def dilateErodeImage(image):
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    dilation = cv2.dilate(image, kernel, iterations = dilationIterations)
    erosion = cv2.erode(dilation,kernel,iterations = erosionIterations)
    return erosion

# find contours
def findContours(image):
    imgray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Contour Approximation 
def approxContours(contours):
    cnt = contours[0]
    epsilon = buildingArcLength * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return approx

# calculate area and perimeter, then print
def calculateGeometry(image, approx):
    area = cv2.contourArea(approx) * (resolution ** 2)
    rospy.loginfo("Building Area = %f", area)
    perimeter = cv2.arcLength(approx,True) * resolution
    rospy.loginfo("Building Perimeter = %f", perimeter)
    ifttt_details("Building Analysis", area, perimeter)
    return

# true if contour perimeter is below threshold
def is_contour_bad(c):
    perimeter = cv2.arcLength(c, True) * resolution
    return (perimeter < badContPerimeter)

# remove minor contours (e.g. furniture) from map
def removeMinorContours(image, contours):
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    for c in contours:
        if is_contour_bad(c):
            cv2.drawContours(mask, [c], -1, 0, -1)
    image = cv2.bitwise_and(image, image, mask=mask)
    return image

# fill the approx contour polygon
def fillApproxContours(image, approx, contours):
    image2 = image.copy()
    cv2.fillPoly(image2, [approx], (0,255,0), lineType=8, shift=0)
    return image2

# bitwise operations
# Code adapted from OpenCV example (accessed 02/08/17):
# http://docs.opencv.org/trunk/d0/d86/tutorial_py_image_arithmetics.html
def addImages(img1, img2):
    img2 = dilateErodeImage(img2.copy())
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]
    # create a mask and inverse mask of img2
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
    dst = cv2.add(img1_bg,img2_fg)
    img1[0:rows, 0:cols ] = dst
    return img1

#-Room Analysis 'Quad Door' Functions-------------------------------------------

# determine room areas
def getRoomAreas(image, contours):
    cnt = 0
    roomList = list()
    for c in contours:
        epsilon = roomArcLength * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        area = cv2.contourArea(approx) * (resolution ** 2) * roomAreaScale
        perimeter = cv2.arcLength(approx, True) * resolution * roomPerimScale
        cnt += 1
        rospy.loginfo("cnt = %d, area = %f, perimeter = %f", cnt, area, perimeter)   
        # label rooms on map
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,str(cnt),(cx,cy), font, 1,(255,0,0),2)
        roomList.append([cnt, area, perimeter])
    ifttt_details("Room Analysis", roomList, "Room Analysis End")
    return

# returns contours of image
def getContours(img):
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# remove minor contours
def removeNoise(img):
    contours = getContours(img)
    for c in contours:
        perimeter = cv2.arcLength(c, True) * resolution
        if perimeter < badContPerimeter:
            cv2.drawContours(img, [c], -1, (0,0,0), -1)
    return

#  Pixels within 'quad door' quadrilateral changed to black
def fillDoorBlack(img, dq):
    ordList = [list(dq[0]), list(dq[1]), list(dq[2]), list(dq[3])]
    ordList.sort()
    if ordList[0][1] > ordList[1][1]:
        tl = ordList[0]
        bl = ordList[1]
    else:
        tl = ordList[1]  
        bl = ordList[0]  
    if getDist(tl, ordList[2]) > getDist(tl, ordList[3]):
        br = ordList[2]
        tr = ordList[3]
    else:
        br = ordList[3]
        tr = ordList[2]
    array = np.array([[[tl[0], tl[1]]], [[tr[0], tr[1]]], [[br[0], br[1]]], [[bl[0], bl[1]]]])
    cv2.drawContours(img, [array], -1, (0,0,0), -1)
    return

# Cycle through 'quad' doors and change pixels to black
def quadDoorRemoval(img, doorQuads):
    cnt = 0
    for i in doorQuads:
        fillDoorBlack(img, doorQuads[cnt])
        cnt += 1
    return

# Check if 'quad door' candidate quadrilaterals are filled with green
def findValidQuads(img):
    quadList = list(quadSet)
    validQuads = []
    for i in quadList:
        listX = [i[0][0], i[1][0], i[2][0], i[3][0]]
        listY = [i[0][1], i[1][1], i[2][1], i[3][1]]
        cnt = 0
        fillCnt = 0
        for j in range(min(listX), max(listX)):
            for k in range(min(listY), max(listY)):
                if img[k][j].tolist() == greenPx:
                    fillCnt += 1
                cnt += 1
        if cnt != 0 and (float(fillCnt)/float(cnt)) > quadFillRatio:
            validQuads.append(i)
    return validQuads

# Determine if two points are invalid 'quad door' candidates
def invalidPnts(a, b):
    x1, y1 = a.ravel()
    x2, y2 = b.ravel()
    dist = getDist([x1, y1], [x2, y2])
    return (dist < quadDistMin or dist > quadDistMax)

# Add 'quad door' candidate coordinates to global variable quadSet
def addToQuadSet(a, b, c, d):
    tmp = [a[0].tolist(), b[0].tolist(), c[0].tolist(), d[0].tolist()]
    tmp.sort()
    tempTuple = (tuple(tmp[0]), tuple(tmp[1]), tuple(tmp[2]), tuple(tmp[3]))
    quadSet.add(tempTuple)
    return

# determine groups of four corners which are 'quad door' candidates
def identifyInitQuads(corners):
    for a in corners:
        for b in corners:
            if invalidPnts(b,a):
                continue
            for c in corners:
                if invalidPnts(c,a) or invalidPnts(c,b):
                    continue
                for d in corners:
                    if invalidPnts(d,a) or invalidPnts(d,b) or invalidPnts(d,c):
                        continue
                    addToQuadSet(a, b, c, d)
    return

# Shi-Tomasi Corner Detector
def findShiTomasiCorners(img):
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, numCornToDetect, 0.01, 1)
    corners = np.int0(corners)
    return corners

#-Room Analysis 'Pair Door' Functions-------------------------------------------

# Check space to left of 'door pair' is green
def invalidLeft(ptL, grad, img):
    x = ptL[0] - 1
    dist = 0
    cnt = 0
    fillCnt = 0
    while dist < (1/resolution):
        y = ptL[1] - (grad * (ptL[0] - x))
        if img[y][x].tolist() == greenPx:
            fillCnt += 1
        cnt += 1
        x -= 1
        dist = getDist([x, y], [ptL[0], ptL[1]])
    return (float(fillCnt) / float(cnt)) < pairFillRatio

# Check space to right of 'door pair' is green
def invalidRight(ptR, grad, img):
    x = ptR[0] + 1
    dist = 0
    cnt = 0
    fillCnt = 0
    while dist < (1/resolution):
        y = ptR[1] + (grad * (x - ptR[0]))
        if img[y][x].tolist() == greenPx:
            fillCnt += 1
        cnt += 1
        x += 1
        dist = getDist([x, y], [ptR[0], ptR[1]])
    return (float(fillCnt) / float(cnt)) < pairFillRatio

# Check space either side of 'door pair' is green
# When door pair gradient = infinity
def invalidInf(pt1, pt2, img):
    yMax = max(pt1[1], pt2[1])
    yMin = min(pt1[1], pt2[1])
    y1 = yMax + 1
    y2 = yMin - 1
    x = pt1[0]
    dist = 0
    cnt = 0
    fillCnt = 0
    while dist < (1/resolution):
        if img[y1][x].tolist() == greenPx:
            fillCnt += 1
        cnt += 1
        y1 += 1
        dist = y1 - yMax
    if (float(fillCnt) / float(cnt)) < pairFillRatio:
        return True
    dist = 0
    cnt = 0
    fillCnt = 0
    while dist < (1/resolution):
        if img[y2][x].tolist() == greenPx:
            fillCnt += 1
        cnt += 1
        y2 -= 1
        dist = yMin - y2
    return (float(fillCnt) / float(cnt)) < pairFillRatio

# Determine if 'door pair' lie within 'door range' of an opposite wall
# When door pair gradient = infinity
def invalidInfGap(img, ctrPt):
    y = ctrPt[1]
    xMin = ctrPt[0] - pairGap
    xMax = ctrPt[0] + pairGap
    if img[y][xMin].tolist() == greenPx:

        x = xMin - 1
        while img[y][x].tolist() == greenPx:
            dist = getDist([x, y], [ctrPt[0], y])
            if dist > quadDistMax:
                return True 
            x -= 1
        drawBlackLine(img, xMax, y, x, y)
    else:
        x = xMax + 1
        while img[y][x].tolist() == greenPx:
            dist = getDist([x, y], [ctrPt[0], y])
            if dist > quadDistMax:
                return True 
            x += 1
        drawBlackLine(img, xMin, y, x, y)
    return False

# Draw a black line where the door has been detected
def drawBlackLine(img, x1, y1, x2, y2):
    array = np.array([[[x1, y1], [x2, y2]]])
    cv2.drawContours(img, [array], -1, (0,0,0), 6)

# Determine if 'door pair' lie within 'door range' of an opposite wall
# When door pair gradient = 0
def invalidFlatGap(img, ctrPt):
    x = ctrPt[0]
    yMin = ctrPt[1] - pairGap
    yMax = ctrPt[1] + pairGap
    # determine direction of nearest wall  
    if img[yMin][x].tolist() == greenPx:
        y = yMin - 1
        while img[y][x].tolist() == greenPx:
            dist = getDist([x, y], [x, ctrPt[1]])
            # If nearest wall not met within valid distance
            if dist > quadDistMax:
                return True 
            y -= 1
        drawBlackLine(img, x, yMax, x, y)
    else:
        y = yMax + 1
        while img[y][x].tolist() == greenPx:
            dist = getDist([x, y], [x, ctrPt[1]])
            if dist > quadDistMax:
                return True 
            y += 1
        drawBlackLine(img, x, yMin, x, y)
    return False

# Determine if 'door pair' lie within 'door range' of an opposite wall
def invalidGap(img, pt1, pt2, grad, inf):
    ctrPt = [(pt2[0] + pt1[0])/2, (pt2[1] + pt1[1])/2]
    # if gradient = infinity
    if inf == True:
        return invalidInfGap(img, ctrPt)
    # if gradient = 0
    if (pt1[1] - pt2[1]) == 0 or grad == 0:
        return invalidFlatGap(img, ctrPt)
    # otherwise...
    normal = -(1 / grad)
    x1 = ctrPt[0] + pairGap
    x2 = ctrPt[0] - pairGap
    y1 = ctrPt[1] + (normal * (x1 - ctrPt[0]))
    y2 = ctrPt[1] - (normal * (ctrPt[0] - x2)) 
    # determine direction of nearest wall  
    if img[y1][x1].tolist() == greenPx:
        while img[y1][x1].tolist() == greenPx:        
            dist = getDist([x1, y1], [ctrPt[0], ctrPt[1]])
            # If nearest wall not met within valid distance
            if dist > quadDistMax:
                return True 
            x1 += 1
            y1 = ctrPt[1] + (normal * (x1 - ctrPt[0]))
    else:
        while img[y2][x2].tolist() == greenPx:        
            dist = getDist([x2, y2], [ctrPt[0], ctrPt[1]])
            if dist > quadDistMax:
                return True 
            x2 -= 1
            y2 = ctrPt[1] - (normal * (ctrPt[0] - x2)) 
    drawBlackLine(img, x1, y1, x2, y2)
    return False  

# Check pixels around door pair to determine if valid
def invalidColorFill(pt1, pt2, img, inf, grad):
    # Check space either side of 'door pair' is green, continue if not
    if inf == True and invalidInf(pt1, pt2, img):
        return True            
    if inf == False and invalidLeft(pt1, grad, img):
        return True 
    if inf == False and invalidRight(pt2, grad, img):
        return True 
    # Determine if valid gap between door pair and nearest wall
    if invalidGap(img, pt1, pt2, grad, inf):
        return True 
    return False

# Check if 'pair door' candidates match door profile
def findValidPairs(img):
    pairList = list(pairSet)
    validPairs = []
    grad = 0
    inf = False
    for i in pairList:
        pt1 = [i[0][0], i[0][1]]
        pt2 = [i[1][0], i[1][1]]
        # Check gradient != infinity
        if (pt2[0] - pt1[0]) != 0:
            grad = (pt2[1] - pt1[1])/(pt2[0] - pt1[0])
            inf = False
        else:
            inf = True
        # Check pixels around door pair to determine if valid
        if invalidColorFill(pt1, pt2, img, inf, grad):
            continue
        validPairs.append(i)
    return validPairs

# Add 'pair door' candidate coordinates to global variable pairSet
def addToPairSet(a, b):
    tmp = [a[0].tolist(), b[0].tolist()]
    tmp.sort()
    tempTuple = (tuple(tmp[0]), tuple(tmp[1]))
    pairSet.add(tempTuple)
    return

# Determine if two points are invalid 'pair door' candidate
def invalidPair(a, b):
    x1, y1 = a.ravel()
    x2, y2 = b.ravel()
    dist = getDist([x1, y1], [x2, y2])
    return (dist < quadDistMin or dist > (0.6 / resolution))

# determine corner pairs which are 'pair door' candidates
def identifyInitPairs(corners):
    for i in corners:
        for j in corners:
            if invalidPair(i, j):
                continue
            addToPairSet(i, j)
    return

#-Controlling Functions---------------------------------------------------------

# Determine building floor area and perimeter
# Return modified image for room analysis
def buildingAnalysis(img):
    image = img.copy()
    # clean image and calculate building floor area and perimeter
    image = convertImageBandW(image)
    image = dilateErodeImage(image)
    contours = findContours(image)
    approx = approxContours(contours)
    calculateGeometry(image, approx)
    # remove minor contours (furniture) from map
    contoursRemoved = removeMinorContours(image, contours)
    # fill approx poly (green)
    polyFilled = fillApproxContours(image, approx, contours)
    # combine contoursRemoved & polyFilled
    combImage = addImages(polyFilled, contoursRemoved)
    return combImage

# Identify virtual doors on the map and segment into rooms.
# Calculate the area and perimeter of each room
def roomAnalysis(image):
    img = image.copy()
    # Corner detection - find 'quad doors'
    corners = findShiTomasiCorners(img)
    # Identify quad door candidates
    identifyInitQuads(corners)
    doorQuads = findValidQuads(img)
    quadDoorRemoval(img, doorQuads)
    removeNoise(img)
    # Corner detection - find 'pair doors'
    corners = findShiTomasiCorners(img)
    # Identify pair door candidates
    identifyInitPairs(corners)
    doorPairs = findValidPairs(img)
    # Calculate area and perimeter of each room
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations = 1)
    contours = getContours(dilation)
    getRoomAreas(dilation, contours)

def main():
    rospy.init_node('callery_map_analysis')
    file_path = rospy.get_param('~map_file_path')
    image = cv2.imread(file_path)
    combImage = buildingAnalysis(image)
    roomAnalysis(combImage)
    rospy.spin()
    return

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
