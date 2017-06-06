import numpy as np
import cv2

PATH_TO_TEST_IMG = '/home/andi/Desktop/Udacity/CarND-Advanced-Lane-Lines/test_images/'

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, sobel_kernel=3,orient='x', thresh=(0,255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    #cv2.imshow('SobelThresh',binary_output*255)
    # Return the result
    return binary_output*255

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    #scale_factor = np.max(absgraddir)/255
    #absgraddir = (absgraddir/scale_factor).astype(np.uint8)
    #cv2.imshow('absgradir',absgraddir)

    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    #cv2.imshow('DirThresh',binary_output)
    # Return the binary image
    return binary_output

# magnitude threshold. Default values chosen close to suggested optimal
def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    #cv2.imshow('sobelx',sobelx)
    #cv2.imshow('sobely',sobely)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    #cv2.imshow('gradmag',gradmag)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    #binary_output = cv2.cvtColor(binary_output,cv2.COLOR_GRAY2BGR)
    #cv2.imshow('MagThresh',binary_output*255)
    # Return the binary image
    return binary_output*255

# apply all color and gradient thresholds and combine them in a binary mask (255,0)
def combined_gradient_thresholds(img,sobel_thresh = (30,120),magnitude_thresh = (30,120),direction_thresh=(0.8,1.2),display=False):
    ksize=3

    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=sobel_thresh)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=sobel_thresh)
    

    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=magnitude_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=direction_thresh)
    if (display):
        cv2.imshow('SobelThreshX',gradx)
        cv2.imshow('SobelThreshY',grady)
        cv2.imshow('MagThresh',mag_binary)
        cv2.imshow('Dir_thresh',dir_binary)
   
    combined = np.zeros_like(dir_binary)
    combined[((gradx >= 240) & (grady >= 240)) | ((mag_binary >= 240) & (dir_binary >= 240))] = 255

    return combined

# test on images from test-set
def main():
    path_to_img = PATH_TO_TEST_IMG + 'test2.jpg'
    img = cv2.imread(path_to_img)
    cv2.imshow('img',img)

    combined = combined_gradient_thresholds(img)

    cv2.imshow('Combined',combined)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()