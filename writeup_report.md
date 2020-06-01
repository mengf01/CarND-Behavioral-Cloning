# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

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

My model consists of a convolution neural network.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the train_steering_model from comma.ai research paper [https://github.com/commaai/research/blob/master/train_steering_model.py](https://github.com/commaai/research/blob/master/train_steering_model.py)
since it's originally designed to predict steering angles.

The network structure is as followings:
| Layer         		|     Output Shape  |	        	Param				|
|:---------------------:|:----------------------:|:-----------------------:|
|cropping2d_1 (Cropping2D) |  (None, 90, 320, 3) |       0
|lambda_1 (Lambda)          |  (None, 90, 320, 3)  |      0
|conv2d_1 (Conv2D)          |  (None, 23, 80, 16)   |     3088
|elu_1 (ELU)                |  (None, 23, 80, 16)    |    0
|conv2d_2 (Conv2D)         |   (None, 12, 40, 32)    |    12832
|elu_2 (ELU)               |   (None, 12, 40, 32)   |     0
|conv2d_3 (Conv2D)          |  (None, 6, 20, 64)    |     51264
|flatten_1 (Flatten)       |   (None, 7680)         |     0
|dropout_1 (Dropout)       |   (None, 7680)       |       0
|elu_3 (ELU)               |   (None, 7680)       |       0
|dense_1 (Dense)           |   (None, 512)        |       3932672
|dropout_2 (Dropout)       |   (None, 512)        |       0
|elu_4 (ELU)               |   (None, 512)       |        0
|dense_2 (Dense)       |       (None, 1)           |      513

Total params: 4,000,369

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation (20%) set.

I've trained the model (with hyperparameters in model.py) and found that the car got out of track in the first sharp curve in track 1, and also the car was always close to the right lane. I played with more data and used multiple data preprocessing methods, and still failed. (See the below section 3).

I then turned to try the NVidia model introduced in the course, and the car finished track 1 smoothly without leaving the road in the first try! See section 2 below for the model details

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Output Shape  |	        	Param				|
|:---------------------:|:----------------------:|:-----------------------:|
|cropping2d_1 (Cropping2D) |  (None, 90, 320, 3) |       0
|lambda_1 (Lambda)          |  (None, 90, 320, 3)  |      0
|conv2d_1 (Conv2D)           | (None, 43, 158, 24)    |   1824
|activation_1 (Activation)   | (None, 43, 158, 24)   |    0
|conv2d_2 (Conv2D)          |  (None, 20, 77, 36)   |     21636
|activation_2 (Activation)  |  (None, 20, 77, 36)   |     0
|conv2d_3 (Conv2D)          |  (None, 8, 37, 48)    |     43248
|activation_3 (Activation)  |  (None, 8, 37, 48)    |     0
|conv2d_4 (Conv2D)          |  (None, 6, 35, 64)    |     27712
|activation_4 (Activation)   | (None, 6, 35, 64)   |      0
|conv2d_5 (Conv2D)          |  (None, 4, 33, 64)  |       36928
|activation_5 (Activation)  |  (None, 4, 33, 64)  |       0
|flatten_1 (Flatten)        |  (None, 8448)     |         0
|dense_1 (Dense)            |  (None, 100)      |         844900
|activation_6 (Activation)   | (None, 100)      |         0
|dense_2 (Dense)             | (None, 50)       |         5050
|activation_7 (Activation)  |  (None, 50)       |         0
|dense_3 (Dense)             | (None, 10)       |         510
|activation_8 (Activation)  |  (None, 10)      |          0
|dense_4 (Dense)            |  (None, 1)       |          11

Total params: 981,819

#### 3. Creation of the Training Set & Training Process

Garbage in, garbage out. Data is very important. For all the approaches below, I used the following preprocessing methods: image RGB normalization and shuffle.

(1) Approach 1: comma.ai model + course data + center camera only
To start with, I've used the data that the course has provided, 8036 images in total. As mentioned above, I first used the comma.ai CNN. I've started to use only the (8036/3) center camera images, with augmented left-and-right flipped images, so 8036/3*2 images in total. Here is the training process:

Epoch 1/5
201/201 [==============================] - 67s 336ms/step - loss: 0.1016 - val_loss: 0.0181

Epoch 2/5
201/201 [==============================] - 64s 318ms/step - loss: 0.0183 - val_loss: 0.0144

Epoch 3/5
201/201 [==============================] - 64s 320ms/step - loss: 0.0138 - val_loss: 0.0135

Epoch 4/5
201/201 [==============================] - 64s 317ms/step - loss: 0.0116 - val_loss: 0.0127

Epoch 5/5
201/201 [==============================] - 64s 318ms/step - loss: 0.0104 - val_loss: 0.0122

Looks good.

(2) Approach 2: comma.ai model + course data + 3 cameras
Then, besides center camera images, I've tried to use the left camera images and right camera images as well with the 0.2 angle correction in the course. So 8036*2 images in total. Here is the training process:

Epoch 1/5
201/201 [==============================] - 72s 356ms/step - loss: 0.1289 - val_loss: 0.0204

Epoch 2/5
201/201 [==============================] - 67s 335ms/step - loss: 0.0222 - val_loss: 0.0180

Epoch 3/5
201/201 [==============================] - 68s 336ms/step - loss: 0.0178 - val_loss: 0.0163

Epoch 4/5
201/201 [==============================] - 67s 334ms/step - loss: 0.0154 - val_loss: 0.0156

Epoch 5/5
201/201 [==============================] - 67s 333ms/step - loss: 0.0139 - val_loss: 0.0151

With such 3x data, the final loss instead becomes larger. I think it's because the 0.2 correction angle may not be the best value. Since it takes time to train and optimize this magic angle value, I decided to get back to approach 1, use only the center camera.

(3) Approach 3: NVIDIA model + course data + center camera only

As mentioned in section 1 above, the comma.ai model always failed the first sharp corner. So I moved on to use NVIDIA model, which is successful. The training process:

Epoch 1/5
201/201 [==============================] - 69s 344ms/step - loss: 0.0111 - val_loss: 0.0093

Epoch 2/5
201/201 [==============================] - 65s 324ms/step - loss: 0.0097 - val_loss: 0.0088

Epoch 3/5
201/201 [==============================] - 66s 330ms/step - loss: 0.0093 - val_loss: 0.0089

Epoch 4/5
201/201 [==============================] - 66s 330ms/step - loss: 0.0089 - val_loss: 0.0088

Epoch 5/5
201/201 [==============================] - 66s 329ms/step - loss: 0.0086 - val_loss: 0.0086

Perfect! The trained model is `output_model/model_nvidia_v1.h5` and the video is `run1.mp4`.

(4) Approach 4: Same as (3) but augmented with my own data:

I drove the car in the simulator for several laps in both clockwise direction and counter-clockwise direction. I also did some "recovery from on-a-lane to center" recordings to enhance the car's ability to get back to center.

Now combining my data and the course data, I have 25920 images in total and I used 25920/3*2 (center camera only, augmented by flipping) to train.

Epoch 1/5
648/648 [==============================] - 284s 438ms/step - loss: 0.0134 - val_loss: 0.0132

Epoch 2/5
648/648 [==============================] - 277s 428ms/step - loss: 0.0120 - val_loss: 0.0122

Epoch 3/5
648/648 [==============================] - 280s 432ms/step - loss: 0.0114 - val_loss: 0.0115

Epoch 4/5
648/648 [==============================] - 277s 427ms/step - loss: 0.0109 - val_loss: 0.0112

Epoch 5/5
648/648 [==============================] - 281s 434ms/step - loss: 0.0103 - val_loss: 0.0105

As you can see, the final loss 0.0105 is larger than that of approach (3), 0.0086. Two possible reasons:
(1) Since this one has more data, I need more epochs to train;
(2) The quality of my own data is not as good as the one the course provided, since I used a "bad" controller (Xbox controller) during training.

Anyway, the car still drive very successfully in the autonomous mode! The trained model is `output_model/model_nvidia_v2.h5` and the video is `run2.mp4`.
