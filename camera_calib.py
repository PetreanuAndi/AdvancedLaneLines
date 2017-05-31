import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

PATH_TO_CALIB_IMG = '/home/andi/Desktop/Udacity/CarND-Advanced-Lane-Lines/camera_cal/'
Test_IMG = '/home/andi/Desktop/Udacity/CarND-Advanced-Lane-Lines/camera_cal/test_image.jpg'
nx=9
ny=6

def rgb2gray(img):
	return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def read_calib_images():
	images = glob.glob(PATH_TO_CALIB_IMG+'calibration*.jpg')
	return images

def getCalibrationStructures():
	objpoints = []
	imgpoints = []

	for file_name in read_calib_images():

		img = cv2.imread(file_name)
		gray = rgb2gray(img)
		#cv2.imshow('gray',gray)
		print('-'*40,file_name)

	

		objp = np.zeros((nx*ny,3),np.float32)
		objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

		ret,corners = cv2.findChessboardCorners(gray,(nx,ny),None)
		if ret == True:
			imgpoints.append(corners)
			objpoints.append(objp)

			#cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
			#cv2.imshow(file_name,img)
			print('*'*20)
		else:
			#cv2.imshow('error',img)
			print('Chessboard Error')

	return objpoints,imgpoints

def performCalibration(img):
	
	gray = rgb2gray(img)
	objpoints,imgpoints = getCalibrationStructures()

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	#dst = cv2.undistort(img, mtx, dist, None, mtx)
	#cv2.imshow('UndistortedTest',dst)

	#cv2.destroyAllWindows()
	return mtx,dist

def main():
	img = cv2.imread(Test_IMG)
	performCalibration(img)
	
	#print(imgpoints)
	#plt.show(img)

if __name__ == '__main__':
	main()
