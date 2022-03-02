<center><h1>MNIST Hand-Written Digit Recognition</h1></center>

### Overview

This notebook uses the k-Nearest Neighbors algorithm to classify hand-written digits from the *MNIST* dataset.
The algorithm is implemented from scratch using vanilla Python. The hyperparameters (k, test-train split ratio)
are then tuned to increase the accuracy of the outputs. Pre-processing for images outside of the MNIST is also
present, and is tested on an image of the number 7 drawn in an image editing program.

## Classification Problem in 2D
<img width="510" alt="image" src="https://user-images.githubusercontent.com/39648391/156468527-ccf845cb-708b-42f3-8c31-6e6da3aaae7a.png">


## Representing Image Data as a Point in 64D Space
<img width="663" alt="image" src="https://user-images.githubusercontent.com/39648391/156468590-b26ad48d-c9ce-453d-a082-982ed66ade8d.png">
<img width="290" alt="image" src="https://user-images.githubusercontent.com/39648391/156468618-0661ccff-3f35-4ce8-9030-733ea77910e7.png">


## Finding the Euclidean Distance in n Dimensions
<img width="543" alt="image" src="https://user-images.githubusercontent.com/39648391/156468492-648a9095-434c-4f6b-b5a5-43b67eb71173.png">


## Classifying MNIST Images
<img width="336" alt="image" src="https://user-images.githubusercontent.com/39648391/156468250-333240ed-3f14-46e7-a4fc-c6bcac5c90dc.png">


## Classifying Custom Images
<img width="429" alt="image" src="https://user-images.githubusercontent.com/39648391/156468165-ce680e17-43cd-48ab-845d-13d3d4869dc7.png">
<img width="324" alt="image" src="https://user-images.githubusercontent.com/39648391/156468388-04204001-252d-4dbf-a92a-efac8a3bda3f.png">


## Tuning the Hyperparameters (k and the test:train split ratio)
<img width="1065" alt="image" src="https://user-images.githubusercontent.com/39648391/156468453-be8be990-f3aa-4f47-a42d-e2986bcfcd05.png">

