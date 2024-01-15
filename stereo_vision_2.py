import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import imutils

import calibration

cap_left = cv2.VideoCapture(2)
cap_right = cv2.VideoCapture(1)


frame_rate = 30 # fps
B = 7.1           # distance between the cameras (cm)
f = 4           # focal length (mm)
alpha = 56.6    # camera field of view in the horisontal plane (degrees)


while (cap_left.isOpened() and cap_right.isOpened()):

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

    
    # If cannot catch any frame, break
    if (succes_right and succes_left):

        

        output_canvas = frame_right
        cv2.imshow("camera (right)", output_canvas)
        cv2.imshow("camera (left)", frame_left)

        # Convert the BGR image to RGB
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)

        # Applying stereo image rectification
        frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)



        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        block_size = 11
        min_disp = -128
        max_disp = 128
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 5
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 200
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 2
        disp12MaxDiff = 0

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )
        disparity_SGBM = stereo.compute(frame_right, frame_left)

        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                beta=0, norm_type=cv2.NORM_MINMAX)
        disparity_SGBM = np.uint8(disparity_SGBM)

        cv2.imshow("disparity_map_SGBM", disparity_SGBM)
        cv2.imwrite("disparity_SGBM_norm.png", disparity_SGBM)

        M = B*f
        # depth_map = M/disparity_SGBM
        # cv2.imshow("depth_map", depth_map)
        # cv2.imshow("right_cam", frame_right)
        # cv2.imshow("left_cam", frame_left)

        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()