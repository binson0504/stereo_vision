import numpy as np
import cv2 as cv
import glob

##### Setup #####

chessboard_size = (9,6)
frame_size = (1280,720)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 27
objp = objp * size_of_chessboard_squares_mm
# print(objp)

objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane
imgpointsR = []

images_left = glob.glob('images\img_left\*.png') #list of images
images_right = glob.glob('images\img_right\*.png')

for img_left, img_right in zip(images_left, images_right):

    #Conver to gray scale
    imgL = cv.imread(img_left)
    imgR = cv.imread(img_right)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboard_size, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if (retL and retR) == True:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL,cornersL, (11,11), (-1,-1), criteria)
        cornersR = cv.cornerSubPix(grayR,cornersR, (11,11), (-1,-1), criteria)
        
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboard_size, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboard_size, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(1000)

cv.destroyAllWindows()



##### Calibration #####

# Finds the camera intrinsic and extrinsic parameters
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
heightL,  widthL = imgL.shape[:2]
heightR,  widthR = imgR.shape[:2]

# Returns the new camera intrinsic matrix based on the free scaling parameter
newcameramtxL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL,heightL), 1, (widthL,heightL))
newcameramtxR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR,heightR), 1, (widthR,heightR))



flags = 0
flags |= cv.CALIB_FIX_INTRINSIC

# perform to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newcameramtxL, distL, newcameramtxR, distR, grayL.shape[::-1], criteria, flags)



# Rectification 
rectifyScale= 1
# Computes rectification transforms for each head of a calibrated stereo camera.
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Parameters saved!")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE) 

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()