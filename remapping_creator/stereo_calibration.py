import numpy as np
import cv2 as cv
import glob

#NOTE this will be run once for the file at the end.

chessBoardSize = (8, 6)
frameSize = (480, 640)

criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

objp = np.zeros((chessBoardSize[0] * chessBoardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessBoardSize[0], 0:chessBoardSize[1]].T.reshape(-1, 2)

objPoints = []
imgPointsL = []
imgPointsR = []

imagesLeft = sorted(glob.glob("images/calibL/*.jpg"))
imagesRight = sorted(glob.glob("images/calibR/*.jpg"))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)

    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    retL, cornersL = cv.findChessboardCorners(grayL, chessBoardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessBoardSize, None)

    if retL and retR == True:
        objPoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        imgPointsL.append(cornersL)
        imgPointsR.append(cornersR)

        cv.drawChessboardCorners(imgL, chessBoardSize, cornersL, retL)
        cv.drawChessboardCorners(imgR, chessBoardSize, cornersR, retR)
        cv.imshow("cornersL", imgL)
        cv.imshow("cornersR", imgR)
        cv.waitKey(1000)

    if retL == False or retR == False:
        print("NOT DETECTED HERE")
        cv.imshow("frameL", imgL)
        cv.imshow("frameR", imgR)
        cv.waitKey(2000)

cv.destroyAllWindows()

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objPoints, imgPointsL, grayL.shape[::-1], None, None)
heightL, widthL, channelsL = imgL.shape

newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objPoints, imgPointsR, grayL.shape[::-1], None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC

critera_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objPoints, imgPointsL, imgPointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], critera_stereo, flags)

rectify_scale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_LF, roi_RF = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectify_scale, (0, 0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("saving parameters...")

cv_file = cv.FileStorage('stereoMapFML3.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereomapL_x', stereoMapL[0])
cv_file.write('stereomapL_y', stereoMapL[1])
cv_file.write('stereomapR_x', stereoMapR[0])
cv_file.write('stereomapR_y', stereoMapR[1])

cv_file.release()
