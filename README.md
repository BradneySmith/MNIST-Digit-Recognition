<center><h1>MNIST Hand-Written Digit Recognition</h1></center>

### Overview

This notebook uses the k-Nearest Neighbors algorithm to classify hand-written digits from the *MNIST* dataset.
The algorithm is implemented from scratch using vanilla Python. The hyperparameters (k, test-train split ratio)
are then tuned to increase the accuracy of the outputs. Pre-processing for images outside of the MNIST is also
present, and is tested on an image of the number 7 drawn in an image editing program.

## Classification Problem in 2D

The k-Nearest Neighbor method is a useful tool for classification problems. Each data point in a set of data can store several pieces of information. In this example, each array stores 64 values, which represent the brightness value of pixels for each pixel in an 8x8 image. The number of values stored can be thought of as the number of dimensions a data point has. 

Thinking about this graphically, a data point with 2 values can be plot in 2-dimensional space. This idea can be taken further to 3-dimensional space for 3 values, and n-dimensional sapce for n values.

Below is an example in 2D, where some data has been plot representing items of one class in orange, and a different class in green. These form clusters since we assume items of the same class have similar properties.

If a point with an unknown class is introduced to the plot, can we determine which class it belongs to? kNN uses the distance to a number of neighboring points to determine the most likely class. You can specify the value of neighbouring points, which generically is called *k*.

To do this, you can find the distance between the new point and every other point on the graph, and sort them in descending order. The modal label in the first *k* results is used to classify the unknown point.

<p align="center">
<img width="510" alt="image" src="https://user-images.githubusercontent.com/39648391/156468527-ccf845cb-708b-42f3-8c31-6e6da3aaae7a.png">
</p>

## Representing Image Data as a Point in 64D Space

The dataset contains 1797 64x1 NumPy arrays, each representing a grid of 8x8 pixels. The value in each element corresponds to the brightness of a pixel, and is a float ranging from 0. (black), to 16. (white), with varying shades of grey in between. Each value can be plot on a coordinate system in 64-dimemsional space, with values representing similar images clustering in similar parts of the space.


<p align="center">
<img width="290" alt="image" src="https://user-images.githubusercontent.com/39648391/156468618-0661ccff-3f35-4ce8-9030-733ea77910e7.png">
</p>
<p align="center">
<img width="663" alt="image" src="https://user-images.githubusercontent.com/39648391/156468590-b26ad48d-c9ce-453d-a082-982ed66ade8d.png">
</p>

## Finding the Euclidean Distance in n Dimensions

The straight-line distance between two points in space is called the Euclidean distance, and for 2-dimensional space can be found using the Pythagorean theorem:

&nbsp;  
<p align="center">
<img width="372" alt="image" src="https://user-images.githubusercontent.com/39648391/156469443-c17a7334-ca0a-48e4-87bf-511a7518e443.png">
</p>
&nbsp;  
&nbsp;  
&nbsp;   
This can be extended to n-dimensional space, where the euclidean distance between 2 points $a = (a_1,a_2,...,a_n)$ and $b = (b_1,b_2,...,b_n)$ is given by

&nbsp;  
<p align="center">
<img width="525" alt="image" src="https://user-images.githubusercontent.com/39648391/156469474-45c7651a-4989-4f60-a673-ce3fea0b3d5a.png">
</p>

## Classifying MNIST Images


## Classifying Custom Images

Now the ```knn``` function has been written and optimised, it can be tested on images not found in the dataset at all. A function has been written to read in every image in a directory called 'images' that is inside the same directory as this notebook. It then loops through each image, reads in the pixel values in greyscale, then resizes the image to a grid of 8x8 pixels. Preparing inputs in this way so that they can be used with an existing algorithm is called 'pre-processing'.

<p align="center">
<img width="429" alt="image" src="https://user-images.githubusercontent.com/39648391/156468165-ce680e17-43cd-48ab-845d-13d3d4869dc7.png">
</p>
<p align="center">
<img width="324" alt="image" src="https://user-images.githubusercontent.com/39648391/156468388-04204001-252d-4dbf-a92a-efac8a3bda3f.png">
</p>


## Tuning the Hyperparameters (k and the test:train split ratio)

One way to find the optimum user-defined parameters (collectively called hyperparameters), is to use a range of combinations and measure the accuracies. In this case, the hyperparameters are *k* and the test-train split ratio, so different combinations of these values parameters can be used to determine the optimum configuration for the algorithm.The next step is to plot the accuracy for each combination, to try to identify any trends. This will help tune algorithm for future classifications.

Some heuristics exist for determining a value for *k* (such as taking the square root of the size of the dataset), but it is common practice to initialise *k* as a random value. Now the algorithm has been tested, its parameters can be tuned.

This algorithm showed best performance with a *k* value of 1. Typically, such small values of *k* are undesirable since they can lead to unstable decision boundaries. Larger values of *k* smooth out decision boundary, making the algorithm more robust to anomylous values. For the training set used, the accuracies for *k* = 1 and *k* = 2 are identical. Since these both produce the highest accuracy, but *k* = 2 has a slightly more stable decision boundary, this value should be taken forward to use with future unknown data points.

The algorithm also showed a non-linear relationship between the test:train split ratio and the accuracy, with accuracy dipping then seeming to peak around 96.7% for a 25:75 split of test:train. This result was more expected, since lower percentages of test data (such as 25%) leave a greater amount of data available to train the algorithm and hence increase its accuracy. The decrease in accuracy with a 10% and 15% split was less expected, and may be attributed to the algorithm 'learning the noise' of the input data set, which would then be detrimental to classifying the testing data

<p align="center">
<img width="1065" alt="image" src="https://user-images.githubusercontent.com/39648391/156468453-be8be990-f3aa-4f47-a42d-e2986bcfcd05.png">
</p>

