import numpy as np
import cv2
from gradient_thresholds import abs_sobel_thresh
from gradient_thresholds import dir_threshold
from gradient_thresholds import mag_thresh
from gradient_thresholds import combined_gradient_thresholds

PATH_TO_TEST_IMG = '/home/andi/Desktop/Udacity/CarND-Advanced-Lane-Lines/test_images/'

# hls S channel mask 
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 255
    return binary_output

# Lab b channel mask
def lab_select(img,thresh = (0,255)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:,:,2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 255
    return binary_output

# Luv L channel mask
def luv_select(img,thresh = (0,255)):
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    l_channel = luv[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 255
    return binary_output

def combined_color_thresholds(img,thresh = (0,255)):
    hls_binary = hls_select(img, thresh)
    #cv2.imshow('SChan',hls_binary)

    lab_binary = lab_select(img, thresh)
    #cv2.imshow('BChan',lab_binary)

    luv_binary = luv_select(img, thresh)
    #cv2.imshow('LChan',luv_binary)

    output = np.zeros_like(luv_binary)
    output[(hls_binary==255)&((lab_binary==255)|(luv_binary==255))] = 255

    #cv2.imshow('Output',output)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return output

# test on images from test-set.
# apply combined gradient-thresholds along with color threshold 
def main():
    path_to_img = PATH_TO_TEST_IMG + 'test5.jpg'
    img = cv2.imread(path_to_img)
    cv2.imshow('img',img)


    hls_binary = hls_select(img, thresh=(170,255))
    cv2.imshow('SChan',hls_binary)

    lab_binary = lab_select(img, thresh=(170,255))
    cv2.imshow('BChan',lab_binary)

    luv_binary = luv_select(img, thresh=(220,255))
    cv2.imshow('LChan',luv_binary)



    #combined_binary = combined_gradient_thresholds(img,sobel_thresh=(40,120),magnitude_thresh = (40,120),direction_thresh=(0.7,1.3))
    #cv2.imshow('GradientCombined',combined_binary)

    #output = np.zeros_like(combined_binary)
    #output[(hls_binary==255)|(combined_binary==255)] = 255

    output = combined_color_thresholds(img,thresh = (190,255))

    cv2.imshow('Output',output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()