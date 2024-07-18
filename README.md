# Computer-Vision-Project

## Parking occupancy detection with traditional computer vision techniques

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UvREmHTqCi1fZ1rsBOegrqWtHjGnYj0h?usp=sharing)

## Project description
With the increasing number of vehicles in urban areas, effective management of parking facilities has become a pressing concern for cities, businesses, and individuals alike. Currently, many parking areas rely on basic sensor-based systems to determine occupancy levels. However, these conventional methods are prone to inaccuracies and often fail to provide real-time data. Advanced parking management solutions offer significant advantages beyond operational efficiency. They facilitate the efficient flow of vehicular traffic, mitigate congestion by directing drivers to alternative parking areas when primary zones reach capacity, and enhance the customer experience by eliminating the need to search for available spaces.

Nowadays, some solutions have been proposed for parking occupancy detection, with most relying on machine learning approaches. The proposed solution, however, involves a pipeline that emphasizes traditional computer vision techniques over newer methodologies.

This emphasis on traditional approaches is motivated by several factors.<br>
Firstly, traditional methods offer a high level of interpretability, providing a clear understanding of the processes involved and the factors influencing results. This is in contrast to some machine learning models, which are characterized by a lack of transparency - the "black box" models.
Conventional methodologies do not necessitate the considerable quantities of labeled data typically required for the training of machine learning models. 
The performance of traditional methods is more predictable than that of machine learning models, which are based on probabilistic models rather than fixed algorithms and rules. This predictability ensures consistent outcomes under defined conditions, reducing the variability and uncertainty often associated with machine learning models.
Additionally, traditional techniques are generally more lightweight and can run efficiently on devices with limited hardware. This is particularly significant for real-world deployment scenarios, where resource constraints may preclude the use of more computationally intensive approaches. 

While it is challenging to achieve state-of-the-art performance with traditional methods, which are often considered outdated, the objective of this project is to come as close as possible to the results obtained by an end-to-end machine learning model.
<br>

## Dataset
The dataset chosen for this task is [CNRPark+EXT](http://cnrpark.it/), which consists in a collection of images obtained from security cameras installed in a parking facility containing 164 parking slots. 
[CNRPark+EXT](http://cnrpark.it/) builds upon an initial dataset known as CNRPark, characterized by a limited number of images, a single weather condition (sunny), and a smaller set of cameras (only 2). However, the extended version, [CNRPark+EXT](http://cnrpark.it/), includes approximately 4000 images captured by 9 distinct cameras operating under three distinct weather conditions (sunny, cloudy, rainy). Furthermore, the dataset includes 9 CSV files, each corresponding to a specific camera, containing the bounding box positions, amounting to a total of 145,000. Each image filename incorporates metadata such as the weather condition, the unique camera identifier, the date, and the time of capture.

In particular, during the implementation of my project, I worked on a subset of the CNRPark+EXT dataset. Specifically, I focused analysis on a set of 1000 images, including 35,250 patches of parking slots. The subset was split into a training set (X_train) and a test set (X_test), with a ratio of 65% to 35%, respectively. This choice was driven by the need to manage the computational resources effectively while implementing a variety of approaches. For each approach, a random search method was developed to identify the optimal parameters.

## Overview of the various approaches that have been implemented. 
<br>

|   Approaches   | Description |
| :---:  |----------- |
| First approach (Adaptive Thresholding)  | It preprocesses images by converting to grayscale, applying Gaussian blur and adaptive thresholding. Morphological operations further enhance features. Predictions of slot occupancy are based on pixel density. | 
| Second Approach (Edges) | V1 involves converting images to grayscale, applying Gaussian blur, and detecting edges using the Canny edge detector.<br>V2 follows the same method as V1 but incorporates morphological operations.<br>V3 also has the same logic as V1, but uses the Laplacian of Gaussian instead of Canny to detect edges.<br>Both versions determine the occupancy status based on the number of detected edges within a cropped patch. |
| Third Approach (Features extraction) | Determines slot occupancy by comparing features to a subset of empty slots. For each slot in the current image, it retrieves patches of the slot when it was known to be empty.  Feature extraction is performed on these patches to create a reference dataset. The same features are extracted from the current image slots. Cosine similarity is used to compare the features from the current image with the reference features. If the similarity exceeds a threshold, the slot is classified as free, otherwise it is classified as occupied.<br>Extracted features include HOG, color histogram, edges count and a combination of these. |
| Fourth Approach (Contour complexity ) | V1 converts images to grayscale, applies adaptive thresholding, and uses median blur.<br>V2 builds on V1 with additional morphological operations and connected component analysis to filter out small isolated regions. For each slot, contours are detected and their complexity is measured using perimeter calculations. Slots with high contour complexity are classified as occupied, while those with low complexity are classified as empty. |
| Fifth Approach (Sift) | V1 SIFT keypoints and descriptors are extracted from the slot's patch. The number of detected keypoints is compared against a threshold to determine occupancy: slots with keypoints above the threshold are classified as occupied, while those below are classified as empty.<br>V2 also uses the SIFT algorithm, but differently. For each slot in the current image, SIFT keypoints and descriptors are extracted. These features are compared against keypoints from images of the same slot when it was empty. If the number of good matches exceeds a specified threshold, the slot is classified as empty; otherwise, it is classified as occupied.
| Sixth Approach (Background Subtraction) |As before, for each slot it identifies free slot images. These free slot images are averaged and a background model is created from them. The current slot image is compared with this background model using background subtraction. The occupancy is determined by the proportion of non-zero pixels in the binary mask. Slots with occupancy above a specified threshold are classified as occupied; otherwise, they are classified as empty. |
| Seventh Approach (Mixed decision) |This approach combines predictions from multiple methods to classify parking slot occupancy. Each method's prediction is weighted by its normalized performance metric (accuracy) on X_train. For each slot, the weighted sum of predictions is calculated. If the weighted average exceeds a threshold, the slot is classified as occupied; otherwise, it is classified as empty.|
|Eighth Approach (Multiple View Point) | This approach incorporates multi-viewpoint data. If the image contains slots observed by other cameras, predictions are made using additional views. The method combines predictions using weighted voting, considering the confidence of each prediction. The confidence is calculated based on the distance of the estimator value from a threshold. If multiple viewpoints are available, predictions are aggregated through weighted majority voting, improving accuracy. |
| Ninth Approaches (Different BB scales)  | The method employs a multi-scale approach to bounding box dimensions, iterating over scaled versions of a single slot image. It utilizes almost every approach implemented, aggregating results from different scales using a majority voting scheme.|
| Tenth Approach (Entropy) | This approach predicts parking slot occupancy by analyzing patches using Fourier entropy. It processes each slot's image to compute Fourier entropy. If the entropy exceeds a threshold, the slot is predicted to be occupied. |


## The performance of the approaches

The evaluation of the approaches is based on the criterion of accuracy, as this was a focus of the studies conducted on the [dataset](http://www.sciencedirect.com/science/article/pii/S095741741630598X). 
|   Approach Name   |   Accuracy   | Accuracy Eight Approach | Accuracy Ninth Approach |
|  :---:   |   :---:   |   :---:   |   :---:   |
| First  | 0.881  | 0.880  |0.885|
| Second V1 | 0.865 | 0.871 | 0.867|
| Second V2 | 0.872  |0.875|0.873|
| Second LoG |0.862|0.864|0.864|
| Third HOG |0.915|0.917|0.921|
| Third COL | 0.647  |0.646|0.648|
| Third EDGE | 0.833  |0.828|0.833|
| Third HOG  COL|0.905|0.897|0.899|
| Third HOG  EDGE|0.917|0.912|0.916|
| Third HOG  COL  EDGE |0.919|0.907|0.914|
| Fourth V1 | 0.822  |0.824|0.830|
| Fourth V2 | 0.794  |0.798|0.795|
| Fifth v1 | 0.840  |0.840|0.840|
| Fifth V2 | 0.785  |  -  |  -  |
| Sixth | 0.849 |0.851| 0.848|
| Seventh | 0.928 |  -  |  -  |
| Tenth | 0.898 |0.899|0.898|

-----------------------
## Tech Used
 ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)  ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
