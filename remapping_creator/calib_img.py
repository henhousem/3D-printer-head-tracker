import numpy as np
import cv2 as cv
import glob


capL = cv.VideoCapture(2)
capR = cv.VideoCapture(0)

img = 0

#NOTE this will be run once for the images so that the stereo calibration file has material
while capL.isOpened and capR.isOpened:

    retL, frameL = capL.read()
    retR, frameR = capR.read()

    k = cv.waitKey(5)

    if k == ord('q'):
        break

    elif k == ord('s'):
        cv.imwrite("images/calibL/imageL" + str(img) + ".jpg", frameL)
        cv.imwrite("images/calibR/imageR" + str(img) + ".jpg", frameR)
        img += 1
    
    cv.imshow('frameL', frameL)
    cv.imshow('frameR', frameR)


capL.release()
capR.release()
cv.destroyAllWindows()
