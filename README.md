# Behavior Cloning

This project is based on Udacity's 
[self-driving car simulator](https://github.com/udacity/self-driving-car-sim).
The point of the code is to train a model using augmented training 
images in a regressive fashion to predict steering data based on 
image data from a car.  The network in use is inspired by 
the network from the NVIDIA paper [End to End Learning for Self-
Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf). 

The idea of the code is to make a pipeline that loads and scales the
images, but allows a Keras generator to apply augmentation for 
each image on a batch basis.  Since the augmentation has aspects
that depends on the image not being normalized (e.g. luminance
modifications), the normalization is done as a Keras Lambda layer
within the model. 

The generator in this code makes use of the following transformations:

 * Luminance to mimic daylight condition changes
 * Flips to increase the training to curve types
 * Horizontal translation to emulate non-centered training data

Other transformations that would be sensible might include:

 * Subtractive luminance banding to emulate shadows
 * Vertical translation to increase road coverage for small data sets
 * Rotations to increase curve training data
 * Stretch and squash to emulate differing road geometries

The subtractive luminance banding would be helpful for generalizing
the training on the brighly lit simulator track to the more advanced 
track with the shadowing.

There are some aspects that are specific to the dynamics of the 
particular simulator.  For example, the degree of steering for a side 
camera versus a centered camera image.  Too little and there is not 
enough compensation to keep the car on the road.  Too much and there
is excessively jerky driving.  These, like other hyperparamters, 
require some fiddling to get to work properly.

This model pre-processes the images on loading them by chopping the 
horizon off and the part with the constant car features near the bottom
of the frame.  These do not help steer the car.  The frame is then
rescaled to 64x64.  This gives enough room to perform max-pooling with
a stride of 2 several times with proper edge considerations, while 
allowing a decent size fully connected network top end.  This has 
the effect of stretching the frame vertically, which actually 
accentuates the curvature of the lane. Soft rectified linear activation
functions appear to work best to help keep the response smooth. A 
reasonable amount of dropout is used to prevent overfitting. An
Adam optimizer is used.  Mean squared error is used since this is 
a regression problem.  A relatively large batch size is used both
to expedite training and to help prevent overfitting.  Normally 
distributed initialization with fan-in + fan-out scaling (Glorot)
is used, as it appears to work well for this problem.

The augmentation is implemented as a Keras generator with variable
batch size.  Depending on the number of zero-steering data points
in the training data, it can be useful to cull some of these
on the fly in the augmentation generator.  For some data I generated
this worked better, but for a the curated data set that is 
available I found it was not needed, as that is a very smooth
data set.  It is easy to add if required, just a conditional on the 
batch being created.  The augmentation is applied and the 
image is normalized in a Keras Lambda layer.

The training data has to be good for the model to work well.  If
you have too jagged of driving, you will have jagged results 
with the cloned behavior.  On my GTX 1060 it takes about two
minutes to train acceptably. `screencast.mp4` is through the
first turn of the track.
