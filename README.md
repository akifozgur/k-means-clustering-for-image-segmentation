## Introduction

### Overview of Image Segmentation
  In the realm of computer vision and image processing, the
concept of image segmentation stands as a critical technique.
Image segmentation involves the partitioning of an image into
distinct and meaningful regions, a process pivotal in extracting
valuable information from visual data. It serves as a fundamental
step, allowing for a more granular understanding and analysis of
images, enabling various downstream applications in fields ranging
from medical imaging to object recognition in autonomous
vehicles.
  The significance of image segmentation lies in its ability to
break down complex visual data into comprehensible units. By
segmenting images into constituent parts based on common
properties, such as color, texture, or intensity, this technique
facilitates the isolation of objects, boundaries, or regions of interest
within an image. This process is pivotal in unlocking a myriad of
possibilities in image analysis and interpretation.

<p align="center"> 
<img src=https://github.com/akifozgur/k-means-clustering-for-image-segmentation/blob/main/img/segmentation.jpg>
</p>

### Overview of Clustering and Its Relevance in Image
Analysis

  One of the prevalent approaches to image segmentation
involves the utilization of clustering algorithms, among which K-
Means clustering stands out as a widely employed technique. K-
Means clustering, a form of unsupervised learning, partitions data
into distinct clusters based on similarity, making it an ideal
candidate for segmenting images into meaningful components.
  This report embarks on a comprehensive exploration of
employing the K-Means clustering algorithm for image
segmentation. The focus is two-fold, encompassing the utilization
of pixel-level features as well as superpixel-level features for
segmentation. Pixel-level features involve extracting information
directly from individual pixels, while superpixel-level features
involve grouping pixels into larger, more coherent units before
feature extraction.

## Overview
In this assignment, I will use the K-means clustering algorithm for image segmentation by using pixel level and superpixel representation of an input image. For this purpose, I will carry out the following steps:
1. Extract feature: I will extract features for input image by using the definitions below.
2. Perform K-means clustering: I will segment image by using extracted features with optimal k parameter.

I'll extract the features listed below and I'll perform the segmentation method by using each of them.

1.  Features
#### Pixel-Level Features
I’ll extract two different features for every pixel in the image; the RGB color feature and spatial location feature.

(a) RGB color feature: I'll concatenate R, G, and B color channel values for each pixel for representation. In other words, each pixel will be represented with [R G B].
(b) RGB color and spatial location feature: Every pixel will be represented with RGB color values and location information which is the coordinate of the pixel. Each pixel will be represented with the [R G B x y] feature vector.

####  Superpixel-Level Features
For this step, I will extract superpixels by using SLIC Superpixel. I will define a feature vector to represent each superpixel.
<p align="center"> 
<img src=https://github.com/akifozgur/k-means-clustering-for-image-segmentation/blob/main/img/superpixel.png>
</p>

(a) Mean of RGB color values: A superpixel is represented with the mean color value of pixels that are included by the superpixel.
(b) RGB color histogram: A superpixel is represented by RGB color histogram which is extracted by using all pixels contained by that super-pixel.
(c) Mean of Gabor filter responses: At this step, I’ll create a filter bank by calculating Gabor filters at different scales and orientations. Then I’ll filter the input image with each Gabor filter. I’ll use the response map to represent superpixels. A superpixel is represented by calculating the mean of Gabor filter response values of pixels that are included by the superpixel.

2. K-Means Clustering
I will implement your own K-Means clustering algorithm for this step. My K-means function will take two parameters; the data matrix(feature matrix/vector) and the k parameter. I’ll use your feature matrix as input data and determine the k parameter which is the count of clusters I want to generate.
I will perform clustering for each feature; RGB color at pixel level, RGB color and location feature at pixel level, mean RGB feature at superpixel level, RGB color histogram at superpixel level, and mean Gabor response at superpixel level.

## How the K-Means Algorithm Works

To process the learning data, the K-means algorithm in data mining starts with a first group of randomly selected centroids, which are used as the beginning points for every cluster, and then performs iterative (repetitive) calculations to optimize the positions of the centroids
It halts creating and optimizing clusters when either:
- The centroids have stabilized — there is no change in their values because the clustering has been successful.
- The defined number of iterations has been achieved.
<p align="center"> 
<img src=https://github.com/akifozgur/k-means-clustering-for-image-segmentation/blob/main/img/kmeans.png>
</p>


## Some Result Images

#### Deer Example

<p align="center"> 
<img src=https://github.com/akifozgur/k-means-clustering-for-image-segmentation/blob/main/img/deer.png>
</p>

#### Soldier Example (RGB Color and Spatial Location Feature)

<p align="center"> 
<img src=https://github.com/akifozgur/k-means-clustering-for-image-segmentation/blob/main/img/soldier.png>
</p>

#### Short Tree Example (Mean of Gabor Filter Responses)

<p align="center"> 
<img src=https://github.com/akifozgur/k-means-clustering-for-image-segmentation/blob/main/img/short_tree.png>
</p>
