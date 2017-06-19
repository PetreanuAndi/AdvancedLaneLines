import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

PATH_TO_CALIB_IMG = '/home/andi/Desktop/Udacity/CarND-Advanced-Lane-Lines/camera_cal/'
Test_IMG = '/home/andi/Desktop/Udacity/CarND-Advanced-Lane-Lines/test_images/straight_lines1.jpg'
nx=9
ny=6

# convert image from bgr to gray
def rgb2gray(img):
	return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# read images with glob
def read_calib_images():
	images = glob.glob(PATH_TO_CALIB_IMG+'calibration*.jpg')
	return images

# find chessboard corners. Save data
def getCalibrationStructures():
	objpoints = []
	imgpoints = []

	for file_name in read_calib_images():

		img = cv2.imread(file_name)
		gray = rgb2gray(img)
		#cv2.imshow('gray',gray)
		#print('-'*40,file_name)

	

		objp = np.zeros((nx*ny,3),np.float32)
		objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

		ret,corners = cv2.findChessboardCorners(gray,(nx,ny),None)
		if ret == True:
			imgpoints.append(corners)
			objpoints.append(objp)

			#cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
			#cv2.imshow(file_name,img)
			#print('*'*20)
		#else:
			#cv2.imshow('error',img)
			#print('Chessboard Error')

	return objpoints,imgpoints

# calibrate camera
def performCalibration(img):
	
	gray = rgb2gray(img)
	objpoints,imgpoints = getCalibrationStructures()

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	dst = cv2.undistort(img, mtx, dist, None, mtx)

	#image = np.hstack((img,dst))
	#cv2.imwrite('./Undistorted.png', image)
	#cv2.imshow('UndistortedTest',image)

	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return mtx,dist

def main():
	img = cv2.imread(Test_IMG)
	performCalibration(img)
	
	#print(imgpoints)
	#plt.show(img)

if __name__ == '__main__':
	main()
