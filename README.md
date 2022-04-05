# Lane-Detection-using-OpenCV

Advanced Lane Detection using Hough Transform.<br>

Finding Radius of Curvature and Turn Prediction of Udacity Lane detection dataset.<br>

Histogram and Adaptive Equalization


### Authors
Rahul Karanam

### Introduction to the Project
The project was divided in three phases as explained below:<br>
1. The first task was to implement histogram and adaptive histogram equalization.<br>
2. The second task was to detect and differntiate straight lane lines.<br> 
3. The third task was to find the radius of curvature and predict the turn prediction.


### Software Required
To run the .py files, use Python 3. Standard Python 3 libraries like OpenCV, Numpy, and matplotlib are used.


### Pipeline for Histogram Equalization
![](https://github.com/karanamrahul/Lane-Detection-using-OpenCV/blob/main/problem1/results/pipeline1/pipeline1.001.jpeg)


### Pipeline for Straight Line Lane Detection
![](https://github.com/karanamrahul/Lane-Detection-using-OpenCV/blob/main/problem2/results/pipeline2/pipeline2.001.jpeg)

### Pipeline for Finding Radius of Curvature and turn Prediction
![](https://github.com/karanamrahul/Lane-Detection-using-OpenCV/blob/main/problem3/output_images/results/pipeline3/pipeline3.001.jpeg)

### Steps to Run the code

#### For Problem 1
To run the code for problem 1a, follow the following commands:

```
cd repository
cd problem1
python3 problem1.py
```
 The above code will output three windows each for histogram equalization,adaptive and clahe.
 
 #### Straight Lane Detection
To run the code for problem 2, follow the following commands:

```
cd repository
cd problem2
python3 problem2.py
```
where the above line will output the detected lanes and it differentiate between the solid and dashed lines.

#### Finding Radius of Curvature and Turn Prediction

To run the code for problem 3 follow the following commands:

```
cd repository
cd problem3
python3 predictTurn.py
```

### Video File Output Links


Output for all the problems: https://drive.google.com/drive/folders/17uvekw6EzbJL63fmnQd51FWaF0KMO2NO?usp=sharing



### References
The following links were helpful for this project:
1. https://github.com/charleswongzx/Advanced-Lane-Lines
2. ENPM 673, Robotics Perception Theory behind Homography Estimation Supplementary Reference
3. https://www.learnopencv.com/tag/projection-matrix/
4. https://www.learnopencv.com/tag/calibration-camera/
