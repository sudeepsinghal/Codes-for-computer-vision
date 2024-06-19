# importing required libs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import numpy as np
import keras
# specifying path of images and binary masks
import tensorflow as tf


image_dir = 'C:/python/projects/pythonProject/DeepGlobe_dataset/DeepGlobe dataset'
mask_dir = 'C:/python/projects/pythonProject/DeepGlobe_dataset/DeepGlobe dataset masks'

#  walking thru the image directory to resize , normalize and store the images as numpy array to a list

image_dataset = []
for path, subdirs, files in os.walk(image_dir):
    # print(path)
    # print(subdirs)
    # print(files)
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'train':
        images = os.listdir(path)
        for i,image_name in enumerate(images):
            if image_name.endswith('jpg'):
                # print(image_name)
                image = cv2.imread(path + '/' + image_name,1)
                image = Image.fromarray(image)
                image = image.resize((400,400))
                image = np.array(image)
                image = image.astype('float32') / 255
                image_dataset.append(image)

print(len(image_dataset))
print(type(image_dataset))

# converting the list to a numpy array as well
image_dataset = np.array(image_dataset)

print(type(image_dataset))

mask_dataset = []
for path, subdirs, files in os.walk(mask_dir):
    # print(path)
    # print(subdirs)
    # print(files)
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'modified_masks':
        mask_images = os.listdir(path)
        for i,mask_name in enumerate(mask_images):
            if mask_name.endswith('png'):
                # print(mask_name)
                mask = cv2.imread(path + '/' + mask_name,1)
                # default the cv2 lib reads it in bgr hence yellow will not appear yellow
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                # mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
                mask = Image.fromarray(mask)
                mask = mask.resize((400,400))
                mask = np.array(mask)
                mask = mask.astype('int32') / 255
                mask_dataset.append(mask)

print(len(mask_dataset))
print(type(mask_dataset))


#converting the list to a numpy array as well
mask_dataset = np.array(mask_dataset)
print(mask_dataset.shape)
print(type(mask_dataset))
import random
# visualizing the images and its corresponding masks to check if they are alligned or not
# for m in range(0,3):
#
#
#     image_number = random.randint(0, len(image_dataset))
#     # print(mask_dataset[image_number])
#     # print(image_dataset[image_number])
#     plt.figure(figsize=(100, 100))
#     plt.subplot(121)
#     plt.imshow(image_dataset[image_number])
#     plt.subplot(122)
#     plt.imshow(mask_dataset[image_number])
#     plt.show()

# indicating what colour represents what area

agri_land = np.array((1,1,1))
other_land = np.array((0,0,0))

# assigning a dummy value
label = mask
# print(label.shape)
# print("mask is " , (label))

# coverting rgb label to 2d labels
def rgb_to_2d(labels):
    label_seg = np.zeros(labels.shape[:-1], dtype=np.uint8)
    label_seg[np.all(labels == (1, 1, 1), axis=-1)] = 1  # Agri land
    label_seg[np.all(labels == (0, 0, 0), axis=-1)] = 0  # Other land
    return label_seg

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2d(mask_dataset[i])
    labels.append(label)
# print(type(labels))
labels = np.array(labels)
labels = np.expand_dims(labels,axis = 3)
print(labels.shape)

# just to check how many labels i have , should be having 2 [0,1]
print("unique" , np.unique(labels))
# print('label' , np.isin(labels[0],(0,1)))

# one - hot encoding the labels
no_of_classes = len(np.unique(labels))
labels_categorical = keras.utils.to_categorical(labels,no_of_classes)

# print(labels_categorical.shape)
# print(image_dataset[11].shape)
# print(mask_dataset[11].shape)

# splitting data for training and testing
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(image_dataset,labels_categorical,test_size=0.20,random_state=42)
print(y_train.shape)
print(x_test.shape)

#NN

image_height = 400
image_width = 400
image_channel = 3




from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization,  Dropout
from keras import backend as K




def multi_unet_model(n_classes=2, IMG_HEIGHT=400, IMG_WIDTH=400, IMG_CHANNELS=3):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(400, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(400, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(400, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(400, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(400, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(400, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(400, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)  # Original 0.1
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])


    print(model.summary())

    return model
model = multi_unet_model(n_classes=2, IMG_HEIGHT=400, IMG_WIDTH=400, IMG_CHANNELS=3)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
history1 = model.fit(x_train, y_train,
                    batch_size = 32,
                    verbose=1,
                    epochs=10,
                    # validation_data=(x_test, y_test),
                    shuffle=False)

(test_loss , test_acc) = model.evaluate(x_test,y_test)
print("Test accuracy: {} , Test Loss: {} ".format(test_acc , test_loss))

# clearing cache
from keras import backend as K
K.clear_session()

import random
import numpy as np
import random
import numpy as np
test_img_number = random.randint(0, len(x_test))
test_img = x_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]
predicted_img_reversed = 1 - predicted_img
print(ground_truth.shape)
ground_truth = ground_truth[:,:,0]





plt.figure(figsize=(12, 8))
plt.subplot(231)

plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img_reversed)
plt.show()