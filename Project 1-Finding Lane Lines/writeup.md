# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The steps of this project are the following:
* Convert image to Grey Scale
* Make it burr to remove noise
* Detect Edges
* Select ROI
* Find lines in the selected ROI. Then merge those lines to find best 2 lines for right and left lane


[//]: # (Image References)

[image1]: ./test_images_output/solidYellowCurve.jpg

---

### Reflection

### 1. Thought process behind merging multiple lines into lines (for left and right lane).

First, few steps are same as what we learnt in class i.e., to make image grey, remove noise, find edges and then find lines in the selected ROI. After finding all line, I am merging lines in 2 sets (left and right) based on the sign of slopes. I am averaging slopes and biases weighted by the length of the line. The thinking here is to remove the impact of small unaligned lines. Following is one of the image output.

![alt text][image1]


### 2. Shortcomings with current pipeline


While experimenting with video. I found that there are few frames where the lines are going completely off. It may be because of finding false lines after canny edge detection.


### 3. Improvements opportunities

   * There is still some opportunity to tune hyperparameters of canny edge detection, Hough lines and Gaussian blurring.

   * We can try to make ROI dynamic based on certain patterns

   * For videos we can try to smoothen the rate of change of line parameters (m, b)