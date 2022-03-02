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

<img width="510" alt="image" src="https://user-images.githubusercontent.com/39648391/156468527-ccf845cb-708b-42f3-8c31-6e6da3aaae7a.png">


## Representing Image Data as a Point in 64D Space

The dataset contains 1797 64x1 NumPy arrays, each representing a grid of 8x8 pixels. The value in each element corresponds to the brightness of a pixel, and is a float ranging from 0. (black), to 16. (white), with varying shades of grey in between. Each value can be plot on a coordinate system in 64-dimemsional space, with values representing similar images clustering in similar parts of the space.

<img width="663" alt="image" src="https://user-images.githubusercontent.com/39648391/156468590-b26ad48d-c9ce-453d-a082-982ed66ade8d.png">
<img width="290" alt="image" src="https://user-images.githubusercontent.com/39648391/156468618-0661ccff-3f35-4ce8-9030-733ea77910e7.png">


## Finding the Euclidean Distance in n Dimensions

The straight-line distance between two points in space is called the Euclidean distance, and for 2-dimensional space can be found using the Pythagorean theorem:

&nbsp;  
$$
d(a,b) = \sqrt{(a_1-b_1)^2 + (a_2-b_2)^2}
$$
&nbsp;  
&nbsp;  
&nbsp;   
This can be extended to n-dimensional space, where the euclidean distance between 2 points $a = (a_1,a_2,...,a_n)$ and $b = (b_1,b_2,...,b_n)$ is given by

&nbsp;  
$$
d(a,b) = \sqrt{(a_1-b_1)^2 + (a_2-b_2)^2 + ... + (a_n-b_n)^2}
$$


## Classifying MNIST Images
<img width="336" alt="image" src="https://user-images.githubusercontent.com/39648391/156468250-333240ed-3f14-46e7-a4fc-c6bcac5c90dc.png">


## Classifying Custom Images
<img width="429" alt="image" src="https://user-images.githubusercontent.com/39648391/156468165-ce680e17-43cd-48ab-845d-13d3d4869dc7.png">
<img width="324" alt="image" src="https://user-images.githubusercontent.com/39648391/156468388-04204001-252d-4dbf-a92a-efac8a3bda3f.png">


## Tuning the Hyperparameters (k and the test:train split ratio)
<img width="1065" alt="image" src="https://user-images.githubusercontent.com/39648391/156468453-be8be990-f3aa-4f47-a42d-e2986bcfcd05.png">

