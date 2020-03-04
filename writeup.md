# Writeup Of Advanced Lane Lines
---
## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration.png "Calibration Steps"
[image2]: ./output_images/undistorted.png "Road Transformed"
[image3]: ./output_images/binary_grad_color.png "Binary Example"
[image4]: ./output_images/persp_transf_boxes.png "Warp Example"
[image5]: ./output_images/perspective.png "Fit Visual"
[image6]: ./output_images/main_image.png "Output"
[video1]: ./output_videos/project_video.mp4 "Video"  

## Code organization
The code that performs the Advanced Lane Line Finder can be found in the [IPython](./P2.ipynb) notebook. all the code can be grouped into five parts:
1. Required Libraries;
2. Algorithm Parameters;
3. Support Functions;
4. Camera Calibration;
5. Pipeline Functions;
6. Executing Section.

---

### 1. Required Libraries
This section (**cell 1**) contains all the libraries required to run the pipeline.
In particular:
```python
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from shapely.geometry.polygon import Polygon
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```
### 2. Algorithm Parameters
This section (**cell 2**) contains all the parameters used by the algorithms that compose the pipeline and the string used to refers to the input/output folders.

|   Parameter            | Value      |
| :--------------------: | :---------:|
| nx                     | 9          |
| ny                     | 6          |
| ksize                  | 15         |
| mod_thresh             | (20, 100)  |
| dir_thresh             | (0.7, 1.3) |
| s_thresh               | (170, 255) |
| lateral_pixel_offset   | 150        |
| upper_pixel_offset     | 400        |
| upper_pixel_length     | 150        |
| MAX_LONG_LENGTH        | 30         |
| MAX_LAT_LENGTH         | 3.7        |
| PERP_TRANS_LONG_PIXELS | 720        |
| PERP_TRANS_LAT_PIXELS  | 960        |
| poly_degree            | 2          |
| nwindows               | 9          |
| margin                 | 100        |
| minpix                 | 50         |

### 3. Support Functions
This section (**cell 3**) contains all the plotting function used to debug the code.
In particular, there is a specific plot function for each pipeline step.

### 4. Camera Calibration
This section (**cell 4**) contains the function `compute_calibration_coefficients(path, img_prefix)` that computes the camera calibration and distortion coefficients.
This function process all the images with prefix equal to `img_prefix` and extension `.jpg` in the `path` folder. The work-flow steps are:
1. convert to gray scale using `cv2.cvtColor()`
2. look for chessboard corners using `cv2.findChessboardCorners()`
3. if the previous step successfully find the corners in the image, then add them to the `imgpoints` data structure

Once all images have been precessed, the camera calibration and distortion coefficients are computed using `cv2.calibrateCamera()`

![alt text][image1]

### 5. Pipeline Functions
This section contains all the functions needed to compose the pipeline for the Advanced Lane Line Finder.
#### Correcting camera distortion
This function (**cell 6**) corrects the lens distortion using the calibration and distortion coefficients computed by `compute_calibration_coefficients()`.
To demonstrate this step, in the image below it is presented a test image and its undistorted version.

![alt text][image2]

#### Gradient Computations
This section (**cell 8**) is composed by different functions in order to take into account different threshold policies. The main function to compute the gradient components are:
* `abs_sobel_thresh()`: it applies Sobel operator over x or over y basing on input parameter and then applies a two fold bounded threshold on it;
* `mag_thresh()`: it computes x and y gradients and computes the overall gradient magnitude. Then, it applies a two fold bounded threshold on it;
* `dir_threshold()`: it computes x and y gradients and computes the overall gradient direction. Then, it applies a two fold bounded threshold on it;

Io order to combine the different gradient transformations, the following function has been implemented:
* `combine_gradients()`: it uses the previous functions to compute a final image combining all the different gradients functions. In particular it returns **(** ***sobel_x*** **&** ***sobel_y*** **) | (** ***magnitude*** **&** ***direction*** **)**.

#### Color Space Computation
This sections (**cell 9**) contains two functions:
* `color_transformation()`: it computes the HLS transformation and extrapolates the S layer to perform a two fold bounded threshold on it;
* `combine_gradinet_with_color()`: it combines the result of the prevoius funtion with the final gradient transformation. In particular it returns **(** ***gradients*** **|** ***color*** **)**.

To demonstrate this step, in the image below it is presented the results of the gradient transformation, the color transformation and the combination of the two transformations.

![alt text][image3]

#### Perspective Transformation
This section (**cell 7**) contains the function `perspective_transform()` that computes the perspective transformation to obtain a birds-eye view of the road.
To perform this transformation are required two set of points:
* four *source* points on the original image that defines a polygon that contains the two lanes;
* four *destination* points on the final image that defines the final polygon in which the original one will be mapped.
The code that performs the points computation is the following one:
```python
src_raw = [[(img_size[0]+upper_pixel_length)*.5, upper_pixel_offset],
           [img_size[0]-lateral_pixel_offset, img_size[1]],
           [lateral_pixel_offset, img_size[1]],
           [(img_size[0]-upper_pixel_length)*.5, upper_pixel_offset]]    
dst_raw = [[img_size[0]-lateral_pixel_offset, 0],
           [img_size[0]-lateral_pixel_offset, img_size[1]],
           [lateral_pixel_offset, img_size[1]],
           [lateral_pixel_offset, 0]]
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 676, 400      | 1052, 0       |
| 1052, 621     | 1052, 621     |
| 150, 621      | 150, 621      |
| 526, 400      | 150, 0        |


I verified that my perspective transform was working as expected by drawing the `src_raw` and `dst_raw` points onto a test image and its warped counterpart.

![alt text][image4]

#### Find Lane Points
This section (**cell 10**) contains the

* `find_lane_pixels()`: it extract the whole histogram of the image and then it uses the most excited sections of it to start the extraction of the lane pixel coordinates using a sliding window algorithm;
* `fit_poly()`: it uses the results of the previous function to fit a polynomial over the pixel coordiantes and compute the (x,y) coordinates of the lane lines;
* `search_around_poly()`: it extracts the lane lines coordinates from an input image using the information of the previous fitting instead of the histogram of the most excited section. It is useful to optimize the lane finding over a video stream;
* `step_find_lane_lines()`: it combines `find_lane_pixels()` with `fit_poly()` to extract the lane lines coordinates from an image;

To demonstrate the results of this section on a single frame, in the image below it is presented a perspective image (input of this section), its histogram and the obtained lane coordinates plotted over the input image.

![alt text][image5]

#### Compute Curvature
This section (**cell 11**) contains the function `measure_curvature_meters()` that is used to compute the lanes curvature starting from the lane coordinates. This function performs also the transformation from pixel reference frame to cartesian reference frame using information on the standard lane length and road width.
It returns the mean between the two lanes curvature.

#### Determine Ego Position
This section (**cell 12**) contains the function `determine_ego_position()` that is used to compute the position of the vehicle with respect to the midpoint between the two lanes. This function performs also the transformation from pixel reference frame to cartesian reference frame using information on the standard lane length and road width.

### 6. Executing section
This section contains the two pipelines used to process the single frame or the video stream and the respective code to use them.

#### Pipeline Single
The **cell 13** contains the function `advenced_lane_finding_pipeline()` that represents the pipeline used to process a single frame. The main instructions that compose the pipeline are:
```python
# remove distortion from image
undist_img = undistorted_image(img, mtx, dist)

# compute combined gradient binary
combined_grad_img = combine_gradients(undist_img, ksize, mod_thresh, dir_thresh)

# compute color binary
color_img = color_transformation(undist_img, s_thresh)

# combine the two contributions
binary_img, stack_img = combine_gradinet_with_color(combined_grad_img, color_img)

# perspective transform
perspec_img, M, Minv, src_raw, dst_raw = perspective_transform(binary_img)

# find lane points
histogram, lane_point_img, left_fitx, right_fitx, ploty = step_find_lane_lines(perspec_img)

# compute curvature
curvature = measure_curvature_meters(left_fitx, right_fitx, ploty)

# compute ego position
ego_pos = determine_ego_position(img, left_fitx, right_fitx)

# compose the final image
fin_img = draw_result(undist_img, perspec_img, lane_point_img, Minv, left_fitx, right_fitx, ploty, curvature, ego_pos, True)
```

The **cell 14** contains the code necessary to execute this pipeline over all the images contained on the ***test_images*** folder.
The result of the described pipeline applied to a single image is showed below:

![alt text][image6]

---

#### Pipeline Video
The **cell 15** contains a modified version of the previous pipeline and a function used to call this pipeline on the current video frame.
In particular, the pipeline is defined in the function `process_current_frame_pip()`.
To improve performances on the video, the instruction
```python
histogram, lane_point_img, left_fitx, right_fitx, ploty = step_find_lane_lines(perspec_img)
```
has been substituted with
```python
# find lane points
if first_exec:
    histogram, lane_point_img, left_fitx, right_fitx, ploty = step_find_lane_lines(perspec_img)
    first_exec = False
else:
    histogram, lane_point_img, left_fitx, right_fitx, ploty = search_around_poly(perspec_img, prev_left_fit, prev_right_fit)

# update prev fit variables
prev_left_fit = np.polyfit(ploty, left_fitx, poly_degree)
prev_right_fit = np.polyfit(ploty, right_fitx, poly_degree)
```

The **cell 16** contains the code necessary to execute this pipeline over the *project_video.mp4*.\
Here's a [link to my video result](./output_videos/project_video.mp4)

---

## Discussion
### Known Issues
* The threshold used for gradient and color transformation needs to be tuned more properly in order to improve performances;
* The thresholding functions needs to be more robust in handling poorly lighted conditions;
* The perspective transformation needs to be improved with better projection in order to preserve all geometrical properties of lines.

### Open Points
The pipeline needs to take into account robustness more deeply. This could be done via:
* Increase the polynomial degree in order to better manage the lane line shape;
* Check between successive frames to identify bad detections and preserve overall performances.
