import numpy as np 
import cv2

PATH_TO_TEST_IMG = '/home/andi/Desktop/Udacity/CarND-Advanced-Lane-Lines/test_images/'


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_warp_points(imgSize):
	offset = 100
	#yoffset = imgSize[1]/2
	print('ImgSize: ',imgSize)
	src = np.float32([
			[2*offset,imgSize[0]],
			[imgSize[1]/2-offset+30, imgSize[0]/2+offset],
			[imgSize[1]/2+offset-30, imgSize[0]/2+offset],
			[imgSize[1]-offset/2,imgSize[0]]
		])
	dst = np.float32([
			[offset,imgSize[0]],
			[offset,0],
			[imgSize[1]-offset,0],
			[imgSize[1]-offset,imgSize[0]]
		])

	return src,dst

def get_inverse_warp(img):
	imshape = img.shape
	imgSize = (imshape[0],imshape[1])
	src,dst = get_warp_points(imgSize)

	Minv = cv2.getPerspectiveTransform(dst,src)
	return Minv

def get_warped_image(img):

	imshape = img.shape
	imgSize = (imshape[0],imshape[1])
	src,dst = get_warp_points(imgSize)
	#print(src)
	#print(dst)

	M = cv2.getPerspectiveTransform(src,dst)
	warped = cv2.warpPerspective(img,M,(imshape[1],imshape[0]),flags=cv2.INTER_LINEAR)

	#cv2.imshow('warped',warped)
	return warped

def main():
	path_to_img = PATH_TO_TEST_IMG + 'test1.jpg'
	img = cv2.imread(path_to_img)
	cv2.imshow('img',img)

	warped = get_warped_image(img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()