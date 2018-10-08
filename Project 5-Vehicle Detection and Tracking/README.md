# **Vehicle Detection Project**
___

The steps of this project are the following:

1. Exploration
    1. Read all test file and Vehicle/Non-vehicle images and check the properties of data.
2. Building classifier
    1. Take data from Vehicle/Non-vehicle images and extract following feature to make input(X) any output(y)
        * spacial features
        * HOG
        * color histogram
    2. Use SVM to make classifier and save as pickle
    3. Use Grid search for optimal parameter cross validation
    4. Test accuracy on validation set
3. Detecting Vehicles
    1. Find smaller windows in each frame of different sizes in which we want to find the vehicle.
    2. Resize all windows to the size of trained data and extract features as we did while training.
    3. Make prediction using trained classifier.
    4. Build heatmap
    5. Apply threshold in heatmap to remove false positive and find final bounding box.
4. Video
    1. Make pipeline for predicting vehicle and try on test_video
    2. Run on project video


[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/slide_win.png
[image4]: ./examples/12.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/24.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the lines 70 through 80 of the file called `project_func.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images and extracted all features.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=8` and `cells_per_block=2`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, but these parameters the accuracy of validation set is maximum.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

First I have normalized the features using sklearn preprocessing `StandardScaler()`. Then used `LinearSVC` to build classifier in cell 7th cell of `Final Project.ipynb`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have decided to for different size of windows I have chosen size from 32 to 160 pixel square. Based on the size of square I have decide in which region to convolve. I have started from the area where road is starting and only bigger windows will be captured for full bottom half. The code is in `pipeline()` function in 11th cell of `Final Project.ipynb`. Here is the image for the same.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I have also iterated on overlap percentage of windows and heatmap threshold to fine tune the pipeline. Here are some example images:

![alt text][image4]
![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

### Here is the test image and its corresponding heatmap:

![alt text][image5]


### Here the resulting bounding boxes are drawn onto the test image:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The implememtation is very slow. For ~1 minute video it took around 2 hours. So this cannot be deployed in realtime. To improve this I will use 1 time hog feature extraction and then slice windows.

We can also use CNN or capsule net for classifier. It can be more accuarte and can also run faster if we use GPU.

We can also use lare data set for training.

