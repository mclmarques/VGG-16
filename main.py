from __future__ import print_function

import os

import cv2
import numpy as np
import pandas as pd
from keras.api.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
from keras.api.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle


def img_classification_vgg(input_tensor=None, classes=2):
    img_rows, img_cols = 224, 224  # by default size is 224,224
    img_channels = 3

    img_dim = (img_rows, img_cols, img_channels)

    img_input = Input(shape=img_dim)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=img_input, outputs=x, name='VGGParkingRecognition')
    return model


def main():
    # Create & compile model
    model = img_classification_vgg(classes=2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Scans for how many classes there are, and how many elements are on each class
    dataset_path = os.listdir('Training_Data')
    data_classes = os.listdir('Training_Data')
    print(data_classes)  # what kinds of classes are in this dataset

    elements = []

    for classType in data_classes:
        # Get count of all the file names
        element_count = os.listdir('Training_Data' + '/' + classType)

        # Add them to the list
        for element in element_count:
            elements.append((classType, str('Training_Data' + '/' + classType) + '/' + element))
            print(elements)

    # Build a dataframe
    elements_data_frame = pd.DataFrame(data=elements, columns=['element type', 'image'])
    print(elements_data_frame.head())
    # print(rooms_df.tail())

    # Checks pre-fetching of data went correct by displaying what was found. This can be removed if not needed
    print("Total number of elements in the dataset: ", len(elements_data_frame))

    elements_count = elements_data_frame['element type'].value_counts()

    print("items in each category: ")
    print(elements_count)

    # Load images & pre-processing images
    im_size = 224
    base_path = "Training_Data/"
    images = []
    labels = []
    for i in data_classes:
        image_path = base_path + str(i)
        filenames = [i for i in os.listdir(image_path)]

        for f in filenames:
            img = cv2.imread(image_path + '/' + f)
            img = cv2.resize(img, (im_size, im_size))
            images.append(img)
            labels.append(i)

    images = np.array(images)
    images = images.astype('float32') / 255.0
    images.shape

    y = elements_data_frame['element type'].values

    y_labelencoder = LabelEncoder()
    y = y_labelencoder.fit_transform(y)
    print(y)

    y = y.reshape(-1, 1)
    onehotencoder = OneHotEncoder()  # Converted  scalar output into vector output where the correct class will be 1 and other will be 0
    Y = onehotencoder.fit_transform(y).toarray()
    Y.shape  # (40, 2)

    images, Y = shuffle(images, Y, random_state=1)

    train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=415)

    # inspect the shape of the training and testing.
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    model.fit(train_x, train_y, epochs=10, batch_size=32)


# The entry point for script execution
if __name__ == "__main__":
    main()
