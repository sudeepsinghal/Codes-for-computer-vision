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
import keras.backend as k




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
                image = np.array(image)
                image = cv2.resize(image , (400,400))
                image = image.astype('int32') / 255
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
                mask = cv2.imread(path + '/' + mask_name)
                # default the cv2 lib reads it in bgr hence yellow will not appear yellow
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
                mask = np.array(mask)
                mask = cv2.resize(mask,(400,400),interpolation = cv2.INTER_NEAREST)
                mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
                mask = mask.astype('int32') / 255
                mask_dataset.append(mask)
print(len(mask_dataset))
print(type(mask_dataset))


#converting the list to a numpy array as well
mask_dataset = np.array(mask_dataset)
print(mask_dataset[0].shape)
print(type(mask_dataset))
import random
# visualizing the images and its corresponding masks to check if they are alligned or not
for m in range(0,3):
    image_number = random.randint(0, len(image_dataset))
    # print(mask_dataset[image_number])
    # print(image_dataset[image_number])
    plt.figure(figsize=(25,25))
    plt.subplot(121)
    plt.imshow(image_dataset[image_number])
    plt.subplot(122)
    plt.imshow(mask_dataset[image_number])
#     # plt.show()

# indicating what colour represents what area

agri_land = np.array((1,1,1))
other_land = np.array((0,0,0))

# assigning a dummy value
label = mask

# coverting rgb label to 2d labels
def rgb_to_2d(labels):
    label_seg = np.zeros(labels.shape[:-1], dtype=np.uint8)
    label_seg[np.all(labels == 1, axis=-1)] = 1  # Agri land
    label_seg[np.all(labels == 0, axis=-1)] = 0  # Other land
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
x_train , x_test, y_train, y_test = train_test_split(image_dataset,labels_categorical,test_size=0.20)
print(y_train.shape)
print(x_test.shape)

#NN

image_height = 400
image_width = 400
image_channel = 3

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization,  Dropout
from keras import backend as K

def multi_unet_model(n_classes=2, IMG_HEIGHT=None, IMG_WIDTH=None, IMG_CHANNELS=3):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

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

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9) # sigmoid gave worst results

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
model = multi_unet_model(n_classes=2, IMG_HEIGHT=400, IMG_WIDTH=400, IMG_CHANNELS=3)
print(model.summary())
model = keras.models.load_model('C:\python\projects\pythonProject\models\\area segmentation model.h5')


optimizer = keras.optimizers.RMSprop()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# history1 = model.fit(x_train, y_train,
#                     batch_size = 8,
#                     verbose=1,
#                     epochs=50,
#                     # validation_data=(x_test, y_test),
#                     shuffle=False)

(test_loss , test_acc) = model.evaluate(x_test,y_test)
print("Test accuracy: {} , Test Loss: {} ".format(test_acc , test_loss))

#saving the model
if os.path.isfile("C:\python\projects\pythonProject\models\\area segmentation model.h5")is False:
    model.save('C:\python\projects\pythonProject\models\\area segmentation model.h5')
if os.path.isfile('C:\python\projects\pythonProject\\savedweights\\area segmentation model.weights.h5')is False:
    model.save_weights('C:\python\projects\pythonProject\\savedweights\\area segmentation model.weights.h5')


# Prediction
y_pred = model.predict(x_test)
y_pred_max = np.argmax(y_pred,axis = 3)
y_pred_max_rev = 1-y_pred_max
y_test_max= np.argmax(y_test,axis = 3)
y_test_max_rev = 1-y_test_max
print(y_test_max.shape)
print("prediction data>>>>",y_pred_max.shape)

plt.subplot(121)
plt.imshow(y_test_max_rev[44])
plt.subplot(122)
plt.imshow(y_pred_max_rev[44])
plt.show()

from keras import metrics
import random
import numpy as np
import random
import numpy as np


# clearing cache
from keras import backend as K
K.clear_session()


# mean iou
iou = []
for k in range(0,len(y_pred_max_rev)):
    n_classes = 2
    IOU = keras.metrics.MeanIoU(num_classes=n_classes)
    result = IOU(y_test_max_rev[k], y_pred_max_rev[k])
    iou.append(result)
    # print(result)
print(np.mean(iou))

# mae
from tensorflow import keras  # Assuming you're using TensorFlow with Keras

mae = []
for s in range(0, len(y_pred_max_rev)):
  n_classes = 2
  mae_metric = keras.metrics.MeanAbsoluteError()
  mae_metric.update_state(y_pred_max_rev[s], y_test_max_rev[s])
  current_mae = mae_metric.result().numpy()
  mae.append(current_mae)
mean_mae = np.mean(mae)
print("mean MAE : {}".format(mean_mae))
print("Top MAE: {}".format(max(mae)))
print("Lowest MAE: {}".format(min(mae)))
print("Standard Deviation : {}".format(np.std(mae)))

print("shapes")
print(x_test.shape)
print(y_pred_max_rev.shape)
print(y_test_max_rev.shape)
# finding out the area of the segmentation masks
def find_area_of_prediction(y_pred_max_rev):
    area_data = []
    pixel_data = []
    for image in y_pred_max_rev:  # Loop through each image in the array
        pixel_count = 0
        total_area = 0
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                pixel_value = (image[row, col].astype('int32'))
                if pixel_value > 0.8:
                    pixel_count += 1
                total_area = (total_area+pixel_count * 50 * 50)/10000000000
        area_data.append(total_area)
        pixel_data.append(pixel_count)
    return area_data, pixel_data

area_data, pixel_data = find_area_of_prediction(y_pred_max_rev)
print("area for prediction" , area_data)
print('pixels for prediction ' , pixel_data)

def find_area_of_ground_truth(y_test_max_rev):
    area_data2 = []
    pixel_data2 = []
    for mask in y_test_max_rev:  # Loop through each mask in the array
        pixel_count2 = 0
        total_area2 = 0
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                pixel_value = mask[row, col]
                if pixel_value > 0.8:
                    pixel_count2 += 1
                total_area2 = (total_area2 + (pixel_count2 * 50 * 50)) / 10000000000
        area_data2.append(total_area2)
        pixel_data2.append(pixel_count2)
    return area_data2, pixel_data2

area_data2, pixel_data2 = find_area_of_ground_truth(y_test_max_rev)
print("area for ground truth : ", area_data2)
print("pixel for ground truth:" , pixel_data2)



#printing out segmentation masks
import random
import matplotlib.pyplot as plt

# Create a figure with a grid of 10 rows and 3 columns
fig, axs = plt.subplots(5, 3, figsize=(60, 60))

for b in range(5):
    img_no = random.randint(0, len(y_test_max_rev) - 1)

    # Plot the testing image
    axs[b, 0].imshow(x_test[img_no])
    axs[b, 0].set_title('Testing Image',fontsize = 10)
    axs[b, 0].axis('off')

    # Plot the ground truth image
    axs[b, 1].imshow(y_test_max_rev[img_no])
    axs[b, 1].set_title('Testing Label {} - {}'.format(area_data2[b], pixel_data2[b]) , fontsize = 10)
    axs[b, 1].axis('off')

    # Plot the predicted image
    axs[b, 2].imshow(y_pred_max_rev[img_no])
    axs[b, 2].set_title('Prediction on test image {} - {}'.format(area_data[b], pixel_data[b]),fontsize = 10)
    axs[b, 2].axis('off')

# Adjust the layout to prevent overlap
plt.tight_layout()
plt.show()




