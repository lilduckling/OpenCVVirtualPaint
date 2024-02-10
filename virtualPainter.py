import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = 'headers'
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColour = (255, 0, 255)
brushThickness = 15
eraserThickness = 15

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    #find hand landmarks
    img = detector.findHand(img)
    lmList = detector.findPosition(img, draw = False)
    
    if len(lmList) != 0:
        # tip of the index 
        x1, y1 = lmList[8][1:]
        # tip of the middle
        x2, y2 = lmList[12][1:]
        
    #checking which fingers are up
        fingers = detector.fingerUp()
        
    #if selection mode - 2 fingers are up 
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25),
                          drawColour, cv2.FILLED)
            #print('selection mode')
            #checking for the clicking
            
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColour = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColour = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColour = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColour = (0, 0, 0)
                    
            
    #if drawing mode - 1 finger is up 
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1), 15, drawColour, cv2.FILLED)
            #print('Drawing mode')
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColour == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColour, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColour, eraserThickness)
            else:  
                cv2.line(img, (xp, yp), (x1, y1), drawColour, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColour, brushThickness)
            xp, yp = x1, y1
    
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    
    # setting the header image
    img[0:125, 0:1280] = header
    
    
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    
    cv2.imshow('image', img)
    cv2.waitKey(1)