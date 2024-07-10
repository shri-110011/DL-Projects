## Cats and Dogs Image Classifier:

**Dataset Overview:**

Cats and Dogs images dataset from Kaggle. This dataset contains 20000 RGB images

of varying sizes for training and 5000 RGB images of varying sizes for testing.

**Objective:**

To build a CNN model that can predict the class/label from the image.

**Approach:**

1. Extracted the image data from the zip file.

2. Used keras.utils.image_dataset_from_directory() to create the train_dataset, validation_dataset and test_dataset.

3. Found Cat to Dog ratio:

- In train_dataset: 1.01

- In validation_dataset: 0.96

- In test_dataset: 1

  The data is balanced in all 3 datasets.

4. Building the adapted **Alexnet model**:

   The CNN architecture used is as follows:

   **conv layer 1**: model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)))

   **batch normalization layer**: model.add(BatchNormalization())

   **max pool layer 1**: model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

   Output of conv layer = floor((n-k+2p)/s) + 1

   Output of max pool layer = floor((n-k+2p)/s) + 1

   n = 227, s=4, k=11
   **Output of conv layer 1** = 55x55x96

   n = 55, s=2, k=3
   **Output of max pool layer 1** = 27x27x96

   **conv layer 2**: model.add(Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu'))

   **batch normalization layer**: model.add(BatchNormalization())

   **max pool layer 2**: model.add(MaxPooling2D(pool_size= (3,3), strides=(2,2)))

   n = 27, s=1, k=5, p=2
   **Output of conv layer 2** = 27x27x256

   n = 27, s=2, k=3
   **Output of max pool layer 2** = 13x13x256

   **conv layer 3**: model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))

   n = 13, s=1, k=3, p=1

   **Output of conv layer 3** = 13x13x384

   **conv layer 4**: model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))

   n = 13, s=1, k=3, p=1

   **Output of conv layer 4** = 13x13x384

   **conv layer 5**: model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))

   **max pool layer 2**: model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

   n = 13, s=1, k=3, p=1
   **Output of conv layer 5** = 13x13x256

   n = 13, s=2, k=3
   **Output of max pool layer 3** = 6x6x256

   **Flatten layer**:

   **Output of flatten layer is**: 9216

   Fully connected layer has 2 hidden layers with dropout regularization added to each of them and an output layer.

   **hidden layer 1**: model.add(Dense(units=256, activation='relu'))

   model.add(Dropout(0.5))

   **hidden layer 2**: model.add(Dense(units=256, activation='relu'))

   model.add(Dropout(0.5))

   **output layer**: model.add(Dense(units=1, activation='sigmoid'))

5. Did model evaluation and prediction and obtained a **train accuracy of 91.6%**,**validation accuracy of 86.7%** and **test accuracy of 87.06%**.
