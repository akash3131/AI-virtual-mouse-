import cv2
import time
import numpy as np
import HandTrackingModule as htm
import autopy

wCam, hCam = 640,480
frameR =120  #Frame reduction

smoothening = 9
plocX, plocY = 0,0
clocX, clocY = 0,0
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
pTime=0
detector = htm.handDetector(maxhands=1)
wScr, hScr = autopy.screen.size()

while True:
    #1.Find the hand landmark
    success,img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findposition(img)

    #2.Get the tip of index and middle finger
    if len(lmList)!=0:
        x1,y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #print(x1,y1,x2,y2)
        #3.check which finger are up
        fingers = detector.fingersUp()
        print(fingers)
    #4.ONly index fineger: moving mode
        if fingers[1]==1 and fingers[2]==0:

            #5.Convert our coordinates
            cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,0),2)
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            #6.smoothen values
            clocX = plocX+(x3-plocX)/smoothening
            clocY = plocY+(y3-plocY)/smoothening
            #7.Move mouse
            autopy.mouse.move(clocX,clocY)
            cv2.circle(img,(x1,y1),10,(255,0,0),cv2.FILLED)
            plocX,plocY = clocX,clocY
        #8. Both index and middle are up the it is clicking mode
        if fingers[1]==1 and fingers[2]==1:
            # 9.Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),15, (0, 255, 0), cv2.FILLED)
                # 10.click mouse if distance short
                autopy.mouse.click()
        #10.click mouse if distance short
    #11.frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

    #12.display


    cv2.imshow("Image",img)
    cv2.waitKey(1)