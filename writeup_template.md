**Advanced Lane Finding Project**


[//]: # (Image References)

[image1]: ./output_images/original.jpg "Original"
[image2]: ./output_images/1SChan.jpg "S Channel Binary Mask"
[image3]: ./output_images/2GradientCombined.jpg "Combined Sobel Gradients Mask"
[image4]: ./output_images/3CombinedThresholds.jpg "Combined Color+Gradient Thresholds"
[image5]: ./output_images/4Bev.jpg "Birds Eye Viuew of the Combined Mask"
[image6]: ./output_images/Undistorted.png "Undistorted side-by-side comparison"
[image7]: ./output_images/Undistorted-test.png "Undistorted straight lines side-by-side comparison"
[video1]: ./output.mp4 "Output Video"


![alt text][image1]

### Camera Calibration

The code i used for camera calibration can be found inside of "camera_calib.py" file.
For a given image, i use the pre-provided chess-board images to "findChessboardCorners" and then compute the calibration parameters/distorsion coefficients from cv2.calibrateCamera method.
cv2.undistort uses these parameters to yield an undistorted image.

![alt text][image6]
![alt text][image7]

### Pipeline (single images)

"gradient_thresholds.py" file contains all the methods required for computing the gradient masks (sobel on X, sobel on Y, direction threshold, magnitude threshold) and a utility "combine_gradient_thresholds" method that takes all of the above and creates a unified binary mask.
The values for the thresholds were experimentally tweaked, around their suggested values, in order to observe the effect on the binary output.

"color_thresholds.py" file contains all the methods required for computing the color masks ( s channel in hls, b channel in Lab, L channel in Luv),
and a utility "combine_color_thresholds" method that takes all of the above and creates a unified binary mask from color.
I perform Hls_binary & (Lab_binary | Luv binary) logical operation in order to obtain my best result.
The shadow part from the project_video is now properly handled 

After computing a combined_mask, the pipeline extracts the "Birds Eye View" image. (incapsulated in "bev.py" file)
The warp points for source and destination were computed with the help of an offset of 100 pixels (offset/2, +- other experimentally obtained value) in order to maintain paralel lines (as close as possible) in the perpendicular perspective transform (in the context of straight line on the road)

I also used a "region_of_interest" method in order to test the visualisation of the source points used for warping.

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

#### Finding Lane Lines, video processing

The main algorithmic body can be found inside of "advanced_lane_finding.py" file, and is mostly a mirror image of the provided tutorial code.
For each frame i undistort, compute the combined_binary_mask and bev of this mask, and calculate the lane lines in a sliding window approach. 
The mean coordinate of the points inside of this sliding window then feeds into the line fitting.

Also, for a more robust and less time-consuming optimisation,  i only do a "blind search" on the first frame of the video, (keeping the previous (rightFit,leftFit) tuple values) and for the following frames i norrow my lane-search down to an interval surrpounding those previous values. This works well for the project_video.mp4, but needs further implementation when confrunted with scenarios in wich i lose track of my previous values (possibly utilising an aditional time-sliding window and a mean value of those)

The curvature was also computed according to the provided material(line 141 in advanced_lane_finding.py), and is displayed on top of the video (both as a global value and as individual left/right values).I noticed sometimes the left/right values differ by quite a lot, and because that does not seem to be accurate, i only take into consideration yielded values that are close in absolute.

For video processing I used moviepy.editor VideoFileClip library and constructed a pipeline callback method "process_frame(img)". 
I had several failed attepts of building this with cv2 methods, but my ffmpeg instalation seemed to cause issues with the reading of the videoCap.

The "provess_frame(img)" callback follows the previously described pipeline (Compute undistorsion params once ---> undistort -> CombinedColorThresh -> CombinedGradientThresh -> BEV -> findLaneLines -> reproject onto original frame) and returns a stacked view of both original image + lane selection and combined_binary_image + lane selection.

This helped me better understand what my pipeline lacks and how to tune my thresholds for optimal result.
I did not approach the challenging videos (other than viewing results) because i am already very late with my submission.


Here's a [link to my video result](./output.mp4)

---

### Future Work

Improve on thresholding for the binary masks, maybe even use adaptive thresholding depending on some image properties like brightness / road fragmentation (caused by shadows) etc.
Another idea would be to weight each color / gradient masks differently, and only use final pixels that are above a certain confidence threshold.
Construct a time-sliding window for the lane-line-search aglorithm in order to make it more robust.


