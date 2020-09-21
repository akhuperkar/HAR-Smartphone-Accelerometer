[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/akhuperkar/HAR-Smartphone-Accelerometer/master?urlpath=https%3A%2F%2Fgithub.com%2Fakhuperkar%2FHAR-Smartphone-Accelerometer%2Fblob%2Fmaster%2FHAR%2520Using%2520Machine%2520Learning%2520.ipynb)

# Human Activity Recognition from Smartphone Accelerometer Data

>**Author**: Abhijit Khuperkar, Data Scientist    
>**Email**: akhuperkar@yahoo.com  
>**Follow on**: [LinkedIn](https://www.linkedin.com/in/abhijitkhuperkar/) | [Twitter](https://twitter.com/akhuperkar) | [Getpocket](https://getpocket.com/@akhuperkar)   


### Introduction

This project demonstrates how to predict the type of physical activity (e.g., walking, climbing stairs) from tri-axial smartphone accelerometer data using supervised machine learning. Smartphone accelerometers are very precise, and different physical activities give rise to unique patterns of acceleration.  

### Input Data

The input data used for training in this project consists of two files. 

1. The first file, [train_time_series.csv](https://courses.edx.org/assets/courseware/v1/b98039c3648763aae4f153a6ed32f38b/asset-v1:HarvardX+PH526x+1T2020+type@asset+block/train_time_series.csv), contains the raw accelerometer data, which has been collected using the [Beiwe research platform](https://github.com/onnela-lab/beiwe-backend), and it has the following format:

    `timestamp, UTC time, accuracy, x, y, z`

    The time series signals are sampled at 10 Hz (0.1 seconds per sample) and contains total 3744 samples and 3 components. `timestamp` column is the time variable. The last three columns labeled `x`, `y`, and `z` correspond to measurements of linear acceleration along each of the three orthogonal axes.


2. The second file, [train_labels.csv](https://courses.edx.org/assets/courseware/v1/d64e74647423e525bbeb13f2884e9cfa/asset-v1:HarvardX+PH526x+1T2020+type@asset+block/train_labels.csv), contains the activity labels. Different activities have been encoded with integers as follows: 

 - 1 = standing, 
 - 2 = walking, 
 - 3 = stairs down, 
 - 4 = stairs up. 

   The activity labels are sampled at 1 Hz (1 second per sample) and contains 375 samples. Because the accelerometers are sampled at high frequency, the labels in [train_labels.csv](https://courses.edx.org/assets/courseware/v1/d64e74647423e525bbeb13f2884e9cfa/asset-v1:HarvardX+PH526x+1T2020+type@asset+block/train_labels.csv) are only provided for every 10th observation in [train_time_series.csv](https://courses.edx.org/assets/courseware/v1/b98039c3648763aae4f153a6ed32f38b/asset-v1:HarvardX+PH526x+1T2020+type@asset+block/train_time_series.csv).

### Approach

Here, the goal is to classify the four physical activities from smartphone accelerometer signals as accurately as possible. My approach to build a machine learning classifier is as follows:

1. The `training_labels.csv` contain labels for every 10 observations in `training_time_series.csv`. This implies each labeled signal is sampled from 10 signals. 
2. I combined 3 axes components in training and test time series into a 4th component of combined magnitude by taking a square root of summation of their squares: sqrt(x^2+y^2+z^2)
3. From 1 & 2 above, I transformed the `training_time_series` dataframe of shape (3744, 3) into a numpy array of shape (375, 10, 4)
4. From this array, I extracted features for each of the 4 components using frequency transformations e.g. Fast fourier transformation values, Power spectral density values, Autocorrelation
5. I split this array of all features into training and validation array in 80:20 ratio. The training set (300 observations) is used to train the classifier. I randomized the training set to avoid classifier getting biased to a particular pattern in time-series
6. To address the activity class imbalance, I used oversampling with SMOTE on the training set 
7. The validation set (75 observations) is used to get a sense of validation accuracy as an indicator of test accuracy
