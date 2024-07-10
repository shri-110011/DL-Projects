## MNIST Digit Classifier

**Dataset Overview:**
MNIST Digit dataset from Keras. This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

**Objective:**
To build a CNN model that can predict the digit from the image.

**Approach:**
1. Scaled the pixel intensity values for each image to be in the ranage 0 to 1.

2. Build a custom CNN based model with 2 conv layer each followed by a max pooling layer.

The CNN architecture used is as follows:

conv layer 1: Conv2D(filters=16, kernel_size=(3,3), strides=(1, 1), padding='valid', activation='relu')
max pool layer 1: MaxPooling2D((2,2))

Output of conv layer = floor((n-k+2p)/s) + 1
Output of max pool layer = floor((n-k+2p)/s) + 1

n = 28, s=1, k=3
Output of conv layer 1 = 26x26x16

n = 26, s=2, k=2
Output of max pool layer 1 = 13x13x16

conv layer 2: Conv2D(filters=16, kernel_size=(3,3), strides=(1, 1), padding='valid', activation='relu', input_shape=(28,28,1))
max pool layer 2: MaxPooling2D((2,2))

n = 13, s=1, k=3
Output of conv layer 2 = 11x11x16

n = 11, s=2, k=2
Output of max pool layer 2 = 5x5x16

Flatten layer:
Output of flatten layer is: 400

Fully connected layer has 2 hidden layers and an output layer.

hidden layer 1: Dense(units=16, activation='relu')

hidden layer 2: Dense(units=16, activation='relu')

output layer: Dense(units=10, activation='softmax')

3. Did model evaluation and prediction and obtained a **train accuracy of 99.02%**, **validation accuracy of 98.52%** and **test accuracy of 98.62%**.
