# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: center_2018_04_12_02_00_31_195.jpg "Center Driving"
[image3]: right_2018_04_12_01_59_53_597.jpg "Recovery Image"
[image4]: right_2018_04_12_01_59_52_997.jpg "Recovery Image"
[image5]: right_2018_04_12_01_59_55_580.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Flipped Image"
[image7]: FLIPPED_center_2018_04_12_02_00_31_195.jpg "Flipped Center Driving"
[image8]: NvidiaModel.png "Nvidia Model Image"
[image9]: ValidationVisual1.png "Validation Visual"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the Nvidia model discussed in the course and consists of five convolution layers with RELU activation (to introduce nonlinearity), followed by a Flatten and a set of fully-connected layers as show in the image below (model.py lines 84-103). I also included some dropout layers to reduce overfitting.

![alt text][image8]

The data is normalized in the model using a Keras lambda layer (code line 85). And I also use a cropping layer to remove some of the noise from things like trees in the foreground that are not useful for training the model (code line 88).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 96, 99 and 101). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 78-79). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 117). Below is a visualization showing the training vs validation loss.

![alt text][image9]

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, adding images flipped on the y-axis (with the steering angle negated) and also driving on the second track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a very simple model just to get an end-to-end framework working. Then try the LeNet model first to see if it would be sufficient, and finally to based my model on the Nvidia model that was discussed in the course since it is known to work very well.

My first step was to use a convolution neural network model similar to the LeNet model I thought this model might be appropriate because it is known to be a good general convolution neural network model. After trying a few runs in the simulator I felt that the Nvidia model might do a better job, so I ultimated implemented the Nvidia model as my final model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include several dropout layers. After some experimenting I settled on three layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected more data by recovering from the left and right sides of the road on track one and also drove on the second track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 84-103) consisted of a convolution neural network with layers and sizes as shown in the visualization below.

![alt text][image8]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one and a half laps on track one using center lane driving, one lap of recovery driving, driving on the second track, and I also used the data provided as part of the project. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help the model train to not be biased toward driving one particular direction. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image7]


After the collection process, I had 36,018 number of data points. I then preprocessed this data by augmenting each image to be flipped on its y-axis and the steering angle negated.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the validation loss starting to increase with more epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
