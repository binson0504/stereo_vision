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
B = 8           # distance between the cameras (cm)
f = 4           # focal length (mm)
alpha = 56.6    # camera field of view in the horisontal plane (degrees)


while (cap_left.isOpened() and cap_right.isOpened()):

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

    
    # If cannot catch any frame, break
    if (succes_right and succes_left):

        # start = time.time()

        output_canvas = frame_right

        # Convert the BGR image to RGB
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)

        # Applying stereo image rectification
        frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

        #CALCULATING DEPTH 
        # center_right = 0
        # center_left = 0

        
        # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
        # All formulas used to find depth is in video presentaion

        #Problem: mp have two points, so just use them to calculate depth
        # depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

        # cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        # cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        # # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
        # print("Depth: ", str(round(depth,1)))

        
        # stereo = cv2.StereoBM_create()

        # The parameters of StereoBM
        # numDisparities = 16
        # blockSize = 11
        # preFilterType = 1
        # preFilterSize = 5
        # preFilterCap = 31
        # textureThreshold = 30
        # uniquenessRatio = 15
        # speckleRange = 0
        # speckleWindowSize = 0
        # disp12MaxDiff = 0
        # minDisparity = 0

        # Setting the parameters before computing disparity map
        # stereo.setNumDisparities(numDisparities)
        # stereo.setBlockSize(blockSize)
        # stereo.setPreFilterType(preFilterType)
        # stereo.setPreFilterSize(preFilterSize)
        # stereo.setPreFilterCap(preFilterCap)
        # stereo.setTextureThreshold(textureThreshold)
        # stereo.setUniquenessRatio(uniquenessRatio)
        # stereo.setSpeckleRange(speckleRange)
        # stereo.setSpeckleWindowSize(speckleWindowSize)
        # stereo.setDisp12MaxDiff(disp12MaxDiff)
        # stereo.setMinDisparity(minDisparity)
        
        # disparity = stereo.compute(frame_right, frame_left)

        # Converting to float32 
        # disparity = disparity.astype(np.float32)
 
        # Scaling down the disparity values and normalizing them 
        # disparity = (disparity/16.0 - minDisparity)/minDisparity

        # Displaying the disparity map
        # cv2.imshow("disparity",disparity)

        # end = time.time()
        # totalTime = end - start

        # fps = 1 / totalTime
        # #print("FPS: ", fps)

        # cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        # cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   


        # Show the frames
        # cv2.imshow("right", frame_right) 
        # cv2.imshow("left", frame_left)

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
        depth_map = M/disparity_SGBM
        cv2.imshow("depth_map", depth_map)
        cv2.imshow("right_cam", frame_right)
        cv2.imshow("left_cam", frame_left)
        
        # depth_thresh = 100.0 # Threshold for SAFE distance (in cm)
 
        # # Mask to segment regions with depth less than threshold
        # mask = cv2.inRange(depth_map,10,depth_thresh)
        
        # # Check if a significantly large obstacle is present and filter out smaller noisy regions
        # if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:
        
        #     # Contour detection 
        #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        
        #     # Check if detected contour is significantly large (to avoid multiple tiny regions)
        #     if cv2.contourArea(cnts[0]) > 0.01*mask.shape[0]*mask.shape[1]:
            
        #         x,y,w,h = cv2.boundingRect(cnts[0])
            
        #         # finding average depth of region represented by the largest contour 
        #         mask2 = np.zeros_like(mask)
        #         cv2.drawContours(mask2, cnts, 0, (255), -1)
            
        #         # Calculating the average depth of the object closer than the safe distance
        #         depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask2)
                
        #         # Display warning text
        #         cv2.putText(output_canvas, "WARNING !", (x+5,y-40), 1, 2, (0,0,255), 2, 2)
        #         cv2.putText(output_canvas, "Object at", (x+5,y), 1, 2, (100,10,25), 2, 2)
        #         cv2.putText(output_canvas, "%.2f cm"%depth_mean, (x+5,y+40), 1, 2, (100,10,25), 2, 2)
        
        # else:
        #     cv2.putText(output_canvas, "SAFE!", (100,100),1,3,(0,255,0),2,3)
        
        # cv2.imshow('output_canvas',output_canvas)



        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()