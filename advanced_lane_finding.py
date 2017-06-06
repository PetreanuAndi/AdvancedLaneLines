import numpy as np 
import cv2
import PyQt4
import os
#import matplotlib
#matplotlib.use("TkAgg")
import imageio
import matplotlib.pyplot as plt
from gradient_thresholds import combined_gradient_thresholds
from color_thresholds import hls_select
from camera_calib import performCalibration
from bev import get_warped_image
from bev import get_inverse_warp
from moviepy.editor import VideoFileClip
imageio.plugins.ffmpeg.download()

PATH_TO_TEST_IMG = '/home/andi/Desktop/Udacity/CarND-Advanced-Lane-Lines/test_images/'
PATH_TO_VIDEO = '/home/andi/Desktop/Udacity/CarND-Advanced-Lane-Lines/'
PATH_TO_OUTPUT_IMAGES = '/home/andi/Desktop/Udacity/CarND-Advanced-Lane-Lines/output_images/'
DISPLAY=False

distorsion = True
blind_search = True

prev_leftFit = []
prev_rightFit = []


# find lane lines. Code is mostly duplicate of suggested tutorial
def findLaneLines(binary_warped):

	# keep a global previous_(rightFit,leftFit) tuple 
	# use blind_Search only for the first frame. Narrow down search for 
	# the following frames, taking into consideration previous_(rightFit,leftFit) tuple
	global blind_search
	global prev_rightFit
	global prev_leftFit

	#print("Warped shape: ",binary_warped.shape)
	height = binary_warped.shape[0]
	width = binary_warped.shape[1]


	histogram = np.sum(binary_warped[int(height/2):,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	if (blind_search==True):
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = binary_warped.shape[0] - (window+1)*window_height
			win_y_high = binary_warped.shape[0] - window*window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			# Draw the windows on the visualization image
			cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
			cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
			# Identify the nonzero pixels in x and y within the window
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

		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 

		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		

		blind_search=False
		prev_leftFit = left_fit
		prev_rightFit = right_fit

	# narrow down search to previous_(rightFit,leftFit) tuple
	else:
		left_lane_inds = ((nonzerox > (prev_leftFit[0]*(nonzeroy**2) + prev_leftFit[1]*nonzeroy + prev_leftFit[2] - margin)) & (nonzerox < (prev_leftFit[0]*(nonzeroy**2) + prev_leftFit[1]*nonzeroy + prev_leftFit[2] + margin)))
		right_lane_inds = ((nonzerox > (prev_rightFit[0]*(nonzeroy**2) + prev_rightFit[1]*nonzeroy + prev_rightFit[2] - margin)) & (nonzerox < (prev_rightFit[0]*(nonzeroy**2) + prev_rightFit[1]*nonzeroy + prev_rightFit[2] + margin)))

		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]

		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		prev_leftFit = left_fit
		prev_rightFit = right_fit

	# /////////////////////////////////////////////////////////////////////////////////
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	#leftx = leftx[::-1]
	#rightx = rightx[::-1] 
	#print(ploty.shape)
	#print(leftx.shape)

	y_eval = np.max(ploty)
	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	# Now our radius of curvature is in meters
	#print(left_curverad, 'm', right_curverad, 'm')
	curvature = (left_curverad+right_curverad)/2.0
	if (abs(left_curverad-right_curverad)>400):
		curvature_text = '          off reading'
	else:
		curvature_text = '{0:.2f}  | Left : {0:.2f} Right : {0:.2f}'.format(curvature,left_curverad,right_curverad)
	# /////////////////////////////////////////////////////////////////////////////////


	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	window_img = np.zeros_like(out_img)

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	return result,ploty,left_fitx,right_fitx,curvature_text

# callback for every frame-processing
# After initial calibration, undistord each image accordingly
# and apply the combined threshold mask (gradients + S color channel)
# Switch to Birds Eye View and find lane lines from the resulting filtered/warped image	
def process_frame(img):

	global distorsion
	global mtx,dist

	if (distorsion==True):
		print('.'*40)
		print('Computing undistort parameters')
		print('.'*40)
		#get calibration matrix and distorsion coefficients
		mtx,dist = performCalibration(img)
		distorsion=False
		

	undistortedImg = cv2.undistort(img, mtx, dist, None, mtx)
    #undistortedImg=img.copy()
	hls_binary = hls_select(undistortedImg, thresh=(170,255))
    # gradient / direction / magnitude . sobel thresh
	combined_binary = combined_gradient_thresholds(undistortedImg,sobel_thresh=(40,120),magnitude_thresh = (40,120),direction_thresh=(0.7,1.3))
    
    # combined mapping
	threshOut = np.zeros_like(combined_binary)
	threshOut[(hls_binary>=240)|(combined_binary>=240)] = 255

    # Birds Eye View
	warped = get_warped_image(threshOut)

	lane_img,ploty,left_fitx,right_fitx,curvature_text = findLaneLines(warped)

	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 100))
	Minv = get_inverse_warp(color_warp)


	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
	resultColor = cv2.addWeighted(undistortedImg, 1, newwarp, 0.3, 0)
	
	colorBinary = np.dstack((threshOut, threshOut, threshOut))
	#print(colorBinary.shape)
	#print(newwarp.shape)
	#print(undistortedImg.shape)

	resultBinary = colorBinary + newwarp*0.3

	result = np.hstack((resultColor,resultBinary))

	cv2.putText(result,curvature_text,(1000,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),1)

	if (DISPLAY):
		#cv2.imshow('0.img',img)
		#cv2.imshow('1.UndistortedTest',undistortedImg)
		#cv2.imshow('2.SChan',hls_binary)
		#cv2.imshow('3.GradientCombined',combined_binary)
		#cv2.imshow('4.Threshold Output',colorBinary)
		#cv2.imshow('5.BEV',warped)
		#cv2.imshow('6.LaneLines',lane_img)
		cv2.imshow('7.Result',result)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return result

def main():
	path_to_img = PATH_TO_TEST_IMG + 'test5.jpg'
	img = cv2.imread(path_to_img)
	cv2.imwrite(PATH_TO_OUTPUT_IMAGES+'original.jpg',img)

	hls_binary = hls_select(img, thresh=(200,255))
	cv2.imwrite(PATH_TO_OUTPUT_IMAGES+'1SChan.jpg',hls_binary)
	combined_binary = combined_gradient_thresholds(img,sobel_thresh=(30,150),magnitude_thresh = (30,150),direction_thresh=(0.7,1.3))

	cv2.imwrite(PATH_TO_OUTPUT_IMAGES+'2GradientCombined.jpg',combined_binary)

	output = np.zeros_like(combined_binary)
	output[(hls_binary==255)|(combined_binary==255)] = 255

	warped = get_warped_image(output)

	cv2.imwrite(PATH_TO_OUTPUT_IMAGES+'4Bev.jpg',warped)
	cv2.imwrite(PATH_TO_OUTPUT_IMAGES+'3CombinedThresholds.jpg',output)

	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	#return

	path_to_video_file = PATH_TO_VIDEO + 'project_video.mp4'
	path_to_video_output = PATH_TO_VIDEO + 'output3.mp4'
	clip = VideoFileClip(path_to_video_file)

	# for each frame, process accordingly
	out_clip = clip.fl_image(process_frame)

	# save output for viewing results
	out_clip.write_videofile(path_to_video_output,audio=False)

if __name__ == '__main__':
    main()
