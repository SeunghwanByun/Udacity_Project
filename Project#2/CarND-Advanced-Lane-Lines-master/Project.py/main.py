# #importing some useful packages
import os
import sys
# sys.path.append('/home/seunghwan/anaconda3/lib/python3.6/site-packages/cv2.so')
cv2_path = '/home/seunghwan/anaconda3/envs/common/lib/python3.6/site-packages/cv2.cpython-36m-x86_64-linux-gnu.so'
sys.path.append(cv2_path)
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import glob

import cv2
from pyspark import SparkConf, SparkContext

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
# %matplotlib inline

import math

# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

class AdvancedLaneFinder:
    # Member variables : Output Images
    OUT_IMG = 0
    SRC_IMG = 0
    BINARY_IMG = 0
    WARP_IMG = 0
    INV_WARP_IMG = 0
    COMBINED_IMG = 0

    object_point = []
    image_point = []

    ret = 0
    mtx = 0
    dist = 0
    rvecs = 0
    tvecs = 0

    # Member variables
    storedFrameNum = 5

    ListBuffer_left = []
    ListBuffer_right = []

    def __init__(self, path, VideoNumber):
        nx = 9
        ny = 6

        obj_point = np.zeros((nx * ny, 3), np.float32)
        obj_point[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        fnames = glob.glob("./camera_cal/calibration*.jpg")

        for fname in fnames:
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret:
                self.object_point.append(obj_point)
                self.image_point.append(corners)

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.object_point, self.image_point, img.shape[1:], None, None)

        capture = cv2.VideoCapture(path)
        frameIndex = 0
        while True:
            if (capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
                capture.open(path)

            ret, frame = capture.read()

            new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = new_frame.shape[:2]

            self.run(new_frame, frameIndex=frameIndex % self.storedFrameNum, VideoNumber=VideoNumber)

            frameIndex += 1
            

            if frameIndex == 10:
                frameIndex = 0

            image_out = self.OUT_IMG
            image_src = self.SRC_IMG
            image_binary = self.BINARY_IMG
            image_warp = self.WARP_IMG
            image_combined = self.COMBINED_IMG

            image_out = cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)
            image_src = cv2.cvtColor(image_src, cv2.COLOR_RGB2BGR)
            image_binary = cv2.cvtColor(image_binary, cv2.COLOR_RGB2BGR)
            # image_warp = cv2.cvtColor(image_warp, cv2.COLOR_RGB2GRAY)
            image_combined = cv2.cvtColor(image_combined, cv2.COLOR_RGB2BGR)

            image_out = cv2.resize(image_out, dsize=((int)(width / 2), (int)(height / 2)), interpolation=cv2.INTER_AREA)
            image_src = cv2.resize(image_src, dsize=((int)(width / 2), (int)(height / 2)), interpolation=cv2.INTER_AREA)
            image_binary = cv2.resize(image_binary, dsize=((int)(width / 2), (int)(height / 2)),
                                      interpolation=cv2.INTER_AREA)
            image_warp = cv2.resize(image_warp, dsize=((int)(width / 2), (int)(height / 2)),
                                    interpolation=cv2.INTER_AREA)
            image_combined = cv2.resize(image_combined, dsize=((int)(width / 2), (int)(height / 2)),
                                        interpolation=cv2.INTER_AREA)

            cv2.imshow("out", image_out)
            cv2.imshow("src", image_src)
            cv2.imshow("binary", image_binary)
            cv2.imshow("warp", image_warp)
            cv2.imshow("combined", image_combined)
            cv2.moveWindow("combined", 1920, 0)
            cv2.moveWindow("out", 1920 + image_out.shape[1], 0)
            cv2.moveWindow("warp", 1920, 480)
            cv2.moveWindow("binary", 1920 + image_out.shape[1], 480)
            cv2.moveWindow("src", 1920 + image_out.shape[1] * 2, 480)
            cv2.imwrite("out.jpg", image_out)
            cv2.imwrite("src.jpg", image_src)
            cv2.imwrite("binary.jpg", image_binary)
            cv2.imwrite("warp.jpg", image_warp)
            cv2.imwrite("combined.jpg", image_combined)

            cv2.waitKey(3)

        capture.release()
        cv2.destroyAllWindows()

    def run(self, src, frameIndex, VideoNumber):
        height, width, channels = src.shape

        src = self.cal_undistort(src, self.object_point, self.image_point, self.mtx, self.dist)

        # image gray channel
        gray = self.grayscale(src)

        # test various color channel
        # s_channel from hls color space
        hls = self.hlsscale(src)
        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
        test = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        test2 = cv2.cvtColor(src, cv2.COLOR_RGB2LUV)
        test3 = cv2.cvtColor(src, cv2.COLOR_RGB2YCrCb)

        H_channel = hsv[:,:,0]
        S_channel = hsv[:,:,1]
        V_channel = hsv[:,:,2]

        L_channel = test[:,:,0]
        A_channel = test[:,:,1]
        B_channel = test[:,:,2]

        LUV_L_channel = test2[:,:,0]
        LUV_U_channel = test2[:,:,1]
        LUV_V_channel = test2[:,:,2]

        # Threshold for Canny Edge
        low_threshold = 70
        high_threshold = 210

        # Threshold for Sobel Edge
        thresh_min = 20
        thresh_max = 100

        # Sobel X
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines
        scaled_sobelX = np.uint8(511 * abs_sobelx / np.max(abs_sobelx))
        print(np.max(abs_sobelx))


        # plt.imshow(scaled_sobelX, cmap='gray')
        # plt.title("scaled_sobelX")
        # plt.show()

        sobel_mask = np.zeros_like(scaled_sobelX)
        sobel_mask[(scaled_sobelX >= thresh_min) & (scaled_sobelX <= thresh_max)] = 1
        binary_output = np.copy(sobel_mask)

        # Otsu's binary in edge field
        ret1, th_edge = cv2.threshold(scaled_sobelX, 0, 511, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # plt.subplot(1, 2, 1)
        # plt.imshow(binary_output, cmap='gray')
        # plt.title("binary_output")
        # plt.subplot(1, 2, 2)
        # plt.imshow(th_edge, cmap='gray')
        # plt.title("th_edge")
        # plt.show()

        # Canny
        canny = cv2.Canny(gray, 50, 150)

        # Otsu's binary
        ret2, th_color = cv2.threshold(S_channel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Binary
        sxbinary = np.zeros_like(scaled_sobelX)
        sxbinary[(scaled_sobelX >= thresh_min) & (scaled_sobelX <= thresh_max)] = 1

        # Threshold color channel
        s_thresh_min = 170
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        # Threshold white color channel
        gray_thresh_min = 220
        gray_thresh_max = 255
        gray_binary = np.zeros_like(gray)
        gray_binary[(gray >= gray_thresh_min) & (gray <= gray_thresh_max)] = 1

        # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(th_color == 255) | (th_edge == 255)] = 255 #

        # plt.subplot(2, 3, 1)
        # plt.imshow(canny, cmap='gray')
        # plt.title("canny")
        # # plt.show()
        # plt.subplot(2, 3, 2)
        # plt.imshow(binary_output, cmap='gray')
        # plt.title("sobel X")
        # # plt.show()
        # plt.subplot(2, 3, 3)
        # plt.imshow(binary_output2, cmap='gray')
        # plt.title("sobel Y")
        # # plt.show()
        #
        # plt.subplot(2, 3, 4)
        # plt.imshow(s_binary, cmap='gray')
        # plt.title("s_binary")
        #
        # plt.subplot(2, 3, 5)
        # plt.imshow(combined_binary, cmap='gray')
        # plt.title("combined binary")
        # plt.show()

        # Set ROI
        # if project_output, middle1 = (width / 2 - 45, height / 2 + 60), middle2 = (width / 2 + 45, height / 2 + 60)
        # if challenge_output, middle1 = (width / 2 - 30, height / 2 + 80), middle2 = (width / 2 + 30, height / 2 + 80)
        masked_img = 0
        if VideoNumber == 1:
            start = (50, height - 20)
            middle1 = (width / 2 - 45, height / 2 + 60)
            middle2 = (width / 2 + 45, height / 2 + 60)
            end = (width - 50, height - 20)

            pts = np.array([start, middle1, middle2, end], np.int32)
            masked_img = self.region_of_interest(combined_binary, [pts])
        elif VideoNumber == 2:
            start = (100, height)
            middle1 = (width / 2, height / 2 + 80)
            middle2 = (width / 2 + 60, height / 2 + 80)
            end = (width - 100, height)

            pts = np.array([start, middle1, middle2, end], np.int32)
            masked_img = self.region_of_interest(combined_binary, [pts])
        elif VideoNumber == 3:
            start = (50, height)
            middle1 = (width / 2 - 45, height / 2 + 60)
            middle2 = (width / 2 + 45, height / 2 + 60)
            end = (width - 50, height)

            pts = np.array([start, middle1, middle2, end], np.int32)
            masked_img = self.region_of_interest(combined_binary, [pts])

        margin_test = 100
        # cv2.circle(src, (margin_test + 45, height), 2, (255, 0, 0), 20)
        # cv2.circle(src, ((int)(width / 2 ), (int)(height / 2 + 80)), 2, (255, 0, 0), 20)
        # cv2.circle(src, ((int)(width / 2 + 60), (int)(height / 2 + 80)), 2, (255, 0, 0), 20)
        # cv2.circle(src, (width - margin_test, height), 2, (255, 0, 0), 20)
        # Set Points for Top-view
        # Left Up
        # cv2.circle(src, (600, (int)(height * 11 / 18)), 2, (255, 0, 0), 20)
        # Right Up
        # cv2.circle(src, (700, (int)(height * 11 / 18)), 2, (255, 0, 0), 20)
        # Left Down
        # cv2.circle(src, (200, height), 2, (255, 0, 0), 20)
        # Right Down
        # cv2.circle(src, (1200, height), 2, (255, 0, 0), 20)

        # Draw lines
        # cv2.line(src, (600, (int)(height * 11 / 18)), (200, height), (255, 0, 0), 5)
        # cv2.line(src, (700, (int)(height * 11 / 18)), (1200, height), (255, 0, 0), 5)

        # Src Point before perspective transform
        # srcPoint = 0
        # dstPoint = 0
        # if VideoNumber == 1:
        #     srcPoint = np.float32(
        #         [[600, (int)(height * 11 / 18)],
        #          [700, (int)(height * 11 / 18)],
        #          [200, height],
        #          [1200, height]]
        #     )
        #
        #     dstPoint = np.float32(
        #         [[300, 0],
        #          [950, 0],
        #          [300, height],
        #          [950, height]]
        #     )
        # elif VideoNumber == 2:
        #     srcPoint = np.float32(
        #         [[(int)(width / 2), (int)(height / 2 + 80)],
        #          [(int)(width / 2 + 45), (int)(height / 2 + 80)],
        #          [100, height],
        #          [width - 100, height]]
        #     )
        #
        #     dstPoint = np.float32(
        #         [[300, 0],
        #          [950, 0],
        #          [300, height],
        #          [950, height]]
        #     )
        # elif VideoNumber == 3:
        srcPoint = np.float32(
            [[600, (int)(height * 11 / 18)],
             [700, (int)(height * 11 / 18)],
             [200, height],
             [1200, height]]
        )

        dstPoint = np.float32(
            [[300, 0],
             [950, 0],
             [300, height],
             [950, height]]
        )

        # Get Image size
        image_size = (width, height)

        # Get Transform matrix
        M = cv2.getPerspectiveTransform(srcPoint, dstPoint)

        # Warp image
        warped = cv2.warpPerspective(masked_img, M, image_size, flags=cv2.INTER_LINEAR)

        # Get Histogram
        histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((warped, warped, warped))
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Hyperparameters
        nwindows = 9
        margin = 100
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(warped.shape[0] // nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            win_y_low = warped.shape[0] - window_height * (1 + window)
            win_y_high = warped.shape[0] - window_height * window

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the sliding windows on out_img
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 255), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 255), 2)

            # Identify the nonzero pixels in x and within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        ### RANSAC Start
        # Let's implement RANSAC algorithm
        random_num = []
        for extract_three_random_number in range(3):
            random_index = random.randrange(0, len(leftx))
            random_num.append(random_index)

        # print(random_num)

        RANSAC_MATRIX_INPUT = [[leftx[random_num[0]] ** 2, leftx[random_num[0]], 1],
                               [leftx[random_num[1]] ** 2, leftx[random_num[1]], 1],
                               [leftx[random_num[2]] ** 2, leftx[random_num[2]], 1]]
        RANSAC_MATRIX_OUTPUT = [[lefty[random_num[0]]],
                                lefty[random_num[1]],
                                lefty[random_num[2]]]
        ### RANSAC Finish

        # Fit a second order polynomial to each using 'np.polyfit'
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # using Stored parameter
        if len(self.ListBuffer_left) < self.storedFrameNum:
            self.ListBuffer_left.append(left_fit)
        else:
            self.ListBuffer_left[frameIndex] = left_fit

        if len(self.ListBuffer_right) < self.storedFrameNum:
            self.ListBuffer_right.append(right_fit)
        else:
            self.ListBuffer_right[frameIndex] = right_fit

        # Calculate new letf_fit, right_fit
        mean_left0, mean_left1, mean_left2 = 0, 0, 0
        for i in range(len(self.ListBuffer_left)):
            mean_left0 += self.ListBuffer_left[i][0]
            mean_left1 += self.ListBuffer_left[i][1]
            mean_left2 += self.ListBuffer_left[i][2]
        mean_left0 /= len(self.ListBuffer_left)
        mean_left1 /= len(self.ListBuffer_left)
        mean_left2 /= len(self.ListBuffer_left)

        mean_right0, mean_right1, mean_right2 = 0, 0, 0
        for i in range(len(self.ListBuffer_right)):
            mean_right0 += self.ListBuffer_right[i][0]
            mean_right1 += self.ListBuffer_right[i][1]
            mean_right2 += self.ListBuffer_right[i][2]
        mean_right0 /= len(self.ListBuffer_right)
        mean_right1 /= len(self.ListBuffer_right)
        mean_right2 /= len(self.ListBuffer_right)

        # Upgrade left_fit and right_fit
        left_fit[0] = mean_left0
        left_fit[1] = mean_left1
        left_fit[2] = mean_left2

        right_fit[0] = mean_right0
        right_fit[1] = mean_right1
        right_fit[2] = mean_right2

        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            # Avoids an error if 'left' and 'right_fit' are still none or incorrect
            print('The function failed to fit a line')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        warped[lefty, leftx] = 255
        warped[righty, rightx] = 255


        # Plots the left and right polynomials on the lane lines
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')

        int_left_fitx = [int(x) for x in left_fitx]
        int_right_fitx = [int(x) for x in right_fitx]
        int_ploty = [int(y) for y in ploty]

        blank_image = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)

        for y in range(len(ploty)):
            cv2.circle(out_img, (int_left_fitx[y], int_ploty[y]), 3, (0, 255, 0), 20)
            cv2.circle(out_img, (int_right_fitx[y], int_ploty[y]), 3, (0, 255, 0), 20)
            cv2.circle(blank_image, (int_left_fitx[y], int_ploty[y]), 3, (0, 255, 0), 20)
            cv2.circle(blank_image, (int_right_fitx[y], int_ploty[y]), 3, (0, 255, 0), 20)

        # out_img[int_ploty, int_left_fitx] = [0, 255, 0]
        # out_img[int_ploty, int_right_fitx] = [0, 255, 0]

        # plt.imshow(out_img)
        # plt.title("Hello~~")
        # plt.show()

        # Get Inverse Transform matrix
        Minv = cv2.getPerspectiveTransform(dstPoint, srcPoint)

        # Get Inverse Warp image
        inv_warped = cv2.warpPerspective(blank_image, Minv, image_size, flags=cv2.INTER_LINEAR)

        # plt.imshow(inv_warped)
        # plt.title("hello~~2")
        # plt.show()

        combined_image = self.weighted_img(src, inv_warped)

        # plt.imshow(combined_image)
        # plt.title("this is combined final image")
        # plt.show()

        self.OUT_IMG = out_img
        self.SRC_IMG = src
        self.BINARY_IMG = combined_binary
        self.WARP_IMG = warped
        self.INV_WARP_IMG = inv_warped
        self.COMBINED_IMG = combined_image

        # plt.subplot(2, 2, 1)
        # plt.imshow(out_img)
        # plt.title('out_img2')
        # plt.show()
        #         plt.subplot(1, 2, 1)
        # plt.subplot(2, 2, 2)
        # plt.imshow(src)
        # plt.show()
        # plt.subplot(2, 2, 3)
        # plt.imshow(combined_binary)
        # plt.show()

        # plt.subplot(2, 2, 4)
        # plt.imshow(warped)
        # plt.show()

        # plt.imshow(inv_warped)
        # plt.show()

    def cal_undistort(self, img, objpoints, imgpoints, mtx, dist):
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        return undist
    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def hlsscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    def canny(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self, img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img, vertices):
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def weighted_img(self, img, initial_img, alpha=0.8, beta=1., gamma=0.):
        return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


images = os.listdir('/home/seunghwan/Udacity/Project#2/CarND-Advanced-Lane-Lines-master/test_images')

path = '/home/seunghwan/Udacity/Project#2/CarND-Advanced-Lane-Lines-master/test_images/'
for image in images:
    src = mpimg.imread(path + image)
    # AdvancedLaneFinder(src)

video_path = '/home/seunghwan/Udacity/Project#2/CarND-Advanced-Lane-Lines-master/'

project_output = 'project_video.mp4'

challenge_output = 'challenge_video.mp4'

harder_output = 'harder_challenge_video.mp4'

path1 = video_path + project_output
path2 = video_path + challenge_output
path3 = video_path + harder_output

AdvancedLaneFinder(path2, 2)
