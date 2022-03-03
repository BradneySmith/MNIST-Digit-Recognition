<p align="center">
  A Vanilla Python Implementation of k-Nearest Neighbors
</p>


### Overview

This repository contains an implementation of the k-Nearest Neighbors algorithm written in vanilla Python. The code is presented in a few different ways:
* A functional programming implementation: kNN_function.py
* An object-oriented programming implementation: kNN_class.py
* A visual approach using a Jupyter Notebook: Digit Recognition.ipynb

The notebook serves as an extended explanation of the algorithm: how it works, and the motivation behind its use and implementation. The function and class implementations serve as 'libraries' than can be imported to perform simple k-Nearest Neighbor classifications.

Below is an overview of the contents of the Jupyter Notebook, including the testing, and tuning of the hyperparameters. The dataset used for training and testing was the MNIST dataset provided by Sci-Kit Learn. Pre-processing functionality has also been added and tested on digits not inside of the MNIST dataset, so that  any image file containing a hand-written digit can be classified using the code. 


## Classification Problem in 2D

The k-Nearest Neighbor method is a useful tool for classification problems. Each data point in a set of data can store several pieces of information. In this example, each array stores 64 values, which represent the brightness value of pixels for each pixel in an 8x8 image. The number of values stored can be thought of as the number of dimensions a data point has. 

Thinking about this graphically, a data point with 2 values can be plot in 2-dimensional space. This idea can be taken further to 3-dimensional space for 3 values, and n-dimensional space for n values.

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
<img width="587" alt="image" src="https://user-images.githubusercontent.com/39648391/156474110-2d2ccfb7-ba4f-4a6e-afd3-4f2003489c18.png">
</p>

## Finding the Euclidean Distance in n Dimensions

To classify a point in n-dimensional space, the kNN algorithm find the Euclidean (straight-line) distance between the point and every other point in the dataset. The distances are then sorted in descending order, and the modal class in the first k distances is used to classify the point. In 2-dimensional space, the Euclidean distance can be found using the Pythagorean theorem:

<p align="center">
<img width="384" alt="image" src="https://user-images.githubusercontent.com/39648391/156474168-7f05bfbd-8e6f-46d3-9450-2fa8af62cb17.png">
</p>
&nbsp;  
&nbsp;  

This can be extended to n-dimensional space, where the euclidean distance between 2 points is given by:

<p align="center">
<img width="528" alt="image" src="https://user-images.githubusercontent.com/39648391/156474202-c8dff449-26ab-4048-953f-799a5e82d384.png">
</p>

## Classifying MNIST Images

To implement the kNN function, simply call the functions define earlier in the correct order. First, values for k, the new datapoint as a 64x1 array, and the corpus of known data should be given as arguments. Then the sorted list of distances should be found between the new datapoint and the existing data. Next, the modal label can be found to classify the image.

<p align="center">
<img width="536" alt="image" src="https://user-images.githubusercontent.com/39648391/156474289-f9dbbfca-3861-453b-964a-6e9edb5df1d3.png">
</p>
<p align="center">
<img width="373" alt="image" src="https://user-images.githubusercontent.com/39648391/156474319-1555606c-b24c-41d5-b9c7-0b4367613265.png">
</p>
  
## Classifying Custom Images

Now the kNN function has been written and optimised, it can be tested on images not found in the dataset. A function has been written to read in every image in a directory called 'images' that is inside the same directory as this notebook. It then loops through each image, reads in the pixel values in greyscale, then resizes the image to a grid of 8x8 pixels. Preparing inputs in this way so that they can be used with an existing algorithm is called 'pre-processing'.

<p align="center">
<img width="429" alt="image" src="https://user-images.githubusercontent.com/39648391/156468165-ce680e17-43cd-48ab-845d-13d3d4869dc7.png">
</p>
<p align="center">
<img width="387" alt="image" src="https://user-images.githubusercontent.com/39648391/156474394-627b5457-efc3-435e-8d8b-94446904cdf9.png">
</p>


## Tuning the Hyperparameters (k and the test:train split ratio)

One way to find the optimum user-defined parameters (collectively called hyperparameters), is to use a range of combinations and measure the accuracies. In this case, the hyperparameters are *k* and the test-train split ratio, so different combinations of these values parameters can be used to determine the optimum configuration for the algorithm.The next step is to plot the accuracy for each combination, to try to identify any trends. This will help tune algorithm for future classifications.

Some heuristics exist for determining a value for *k* (such as taking the square root of the size of the dataset), but it is common practice to initialise *k* as a random value. Now the algorithm has been tested, its parameters can be tuned.

This algorithm showed best performance with a *k* value of 1. Typically, such small values of *k* are undesirable since they can lead to unstable decision boundaries. Larger values of *k* smooth out decision boundary, making the algorithm more robust to anomylous values. For the training set used, the accuracies for *k* = 1 and *k* = 2 are identical. Since these both produce the highest accuracy, but *k* = 2 has a slightly more stable decision boundary, this value should be taken forward to use with future unknown data points.

The algorithm also showed a non-linear relationship between the test:train split ratio and the accuracy, with accuracy dipping then seeming to peak around 96.7% for a 25:75 split of test:train. This result was more expected, since lower percentages of test data (such as 25%) leave a greater amount of data available to train the algorithm and hence increase its accuracy. The decrease in accuracy with a 10% and 15% split was less expected, and may be attributed to the algorithm 'learning the noise' of the input data set, which would then be detrimental to classifying the testing data.

<p align="center">
<img width="1065" alt="image" src="https://user-images.githubusercontent.com/39648391/156468453-be8be990-f3aa-4f47-a42d-e2986bcfcd05.png">
</p>

