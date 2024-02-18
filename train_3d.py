import keras
import tensorflow as tf
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras import backend as K
from models.Attention_Unet import attention_unet
from models.UNetplusplus import unet_plusplusv2
from models.DCI_Unet import dci_unet
from models.vnet import vnet

# if __name__ =="__main__":
# version and gpu check
print(tf.__version__)
print(keras.__version__)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

def preprocess_input(img):
  return img / 255.0

# Load input images and masks.
# Here we load 256x256x256 pixel volume. We will break it into patches of 64x64x64 for training.
image = io.imread('./3D_dataset/train/train_3D.tif')
img_patches = patchify(image, (64, 64, 64), step=64)  # Step=64 for 64 patches means no overlap

mask = io.imread('./3D_dataset/train/mask_train.tif')
mask_patches = patchify(mask, (64, 64, 64), step=64)
mask_patches = mask_patches / 255
plt.imshow(img_patches[1, 2, 3, :, :, 32])

# input_img and input_mask preparation
input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
input_mask = np.reshape(mask_patches, (-1, mask_patches.shape[3], mask_patches.shape[4], mask_patches.shape[5]))

n_classes = 2

# Convert grey image to 3 channels by copying channel 3 times.
# We do this as our U-Net++v2 model expects 3 channel input.
train_img = np.stack((input_img,) * 3, axis=-1)
train_mask = np.expand_dims(input_mask, axis=4)

train_mask_cat = to_categorical(train_mask, num_classes=n_classes)

X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask_cat, test_size=0.10, random_state=0)


# Loss Function and coefficients to be used during training
def dice_coefficient(y_true, y_pred):
  smoothing_factor = 1
  flat_y_true = K.flatten(y_true)
  flat_y_pred = K.flatten(y_pred)
  return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (
            K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)


def dice_coefficient_loss(y_true, y_pred):
  return 1 - dice_coefficient(y_true, y_pred)


# Model parameters
input_shape = (64, 64, 64, 3)
n_classes = 2
LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)

total_loss = dice_coefficient_loss
metrics = [dice_coefficient]

X_train_prep = X_train / 255.0  # Assuming pixel values are in the range [0, 255]
X_test_prep = X_test / 255.0

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Build and compile the U-Net++v2 model
model = vnet(input_shape, n_classes)   # ['attention_unet', 'unet_plusplusv2', 'dci_unet', 'vnet']
model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())

# Fit the model
history = model.fit(X_train_prep,
                    y_train,
                    batch_size=8,
                    epochs=100,
                    verbose=1,
                    validation_data=(X_test_prep, y_test),
                    callbacks=[early_stopping])

# Save the model for future use
model.save('./pt/vnet.h5')


###
#plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['dice_coefficient']
val_acc = history.history['val_dice_coefficient']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()
#
# #Load the pretrained model for testing and predictions.
# from keras.models import load_model
# my_model = load_model('./pt/vnet.h5', compile=False)
# #If you load a different model do not forget to preprocess accordingly.
#
# #Predict on the test data
# y_pred=my_model.predict(X_test)
# # y_pred_argmax=np.argmax(y_pred, axis=4)
# # y_test_argmax = np.argmax(y_test, axis=4)
# threshold = 0.5
# y_pred_binary = (y_pred > threshold).astype(int)
# y_test_binary = (y_test > threshold).astype(int)
#
# print(y_pred_binary.shape)
# print(y_test_binary.shape)
# print(np.unique(y_pred_binary))
#
# #Using built in keras function for IoU
# #Only works on TF > 2.0
# #from keras.metrics import MeanIoU
# #from keras.metrics import MeanIoU
# #n_classes = 4
# #IOU_keras = MeanIoU(num_classes=n_classes)
# #IOU_keras.update_state(y_test_argmax, y_pred_argmax)
# #print("Mean IoU =", IOU_keras.result().numpy())
#
# #Test some random images
# import random
# test_img_number = random.randint(0, len(X_test))
# test_img = X_test[test_img_number]
# ground_truth=y_test[test_img_number]
#
# test_img_input=np.expand_dims(test_img, 0)
# test_img_input1 = preprocess_input(test_img_input)
#
# test_pred1 = my_model.predict(test_img_input1)
# test_prediction1 = np.argmax(test_pred1, axis=4)[0,:,:,:]
# print(test_prediction1.shape)
# ground_truth_argmax = np.argmax(ground_truth, axis=3)
# print(test_img.shape)
#
#
# #Plot individual slices from test predictions for verification
# slice = 14
# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[slice,:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth_argmax[slice,:,:])
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(test_prediction1[slice,:,:])
# plt.show()
#
# """Now segment the full volume using the trained model."""
#
# #Break the large image (volume) into patches of same size as the training images (patches)
# large_image = io.imread('./3D_dataset/test/test_3D.tif')
# patches = patchify(large_image, (64, 64, 64), step=64)  #Step=256 for 256 patches means no overlap
# print(large_image.shape)
# print(patches.shape)
#
# # Predict each 3D patch
# predicted_patches = []
# for i in range(patches.shape[0]):
#   for j in range(patches.shape[1]):
#     for k in range(patches.shape[2]):
#       #print(i,j,k)
#       single_patch = patches[i,j,k, :,:,:]
#       single_patch_3ch = np.stack((single_patch,)*3, axis=-1)
#       single_patch_3ch_input = preprocess_input(np.expand_dims(single_patch_3ch, axis=0))
#       single_patch_prediction = my_model.predict(single_patch_3ch_input)
#       single_patch_prediction_argmax = np.argmax(single_patch_prediction, axis=4)[0,:,:,:]
#       predicted_patches.append(single_patch_prediction_argmax)
#
# #Convert list to numpy array
# predicted_patches = np.array(predicted_patches)
# print(predicted_patches.shape)
#
# #Reshape to the shape we had after patchifying
# predicted_patches_reshaped = np.reshape(predicted_patches,
#                                         (patches.shape[0], patches.shape[1], patches.shape[2],
#                                          patches.shape[3], patches.shape[4], patches.shape[5]) )
# print(predicted_patches_reshaped.shape)
#
# #Repach individual patches into the orginal volume shape
# reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
# print(reconstructed_image.shape)
#
# print(reconstructed_image.dtype)
#
# #Convert to uint8 so we can open image in most image viewing software packages
# reconstructed_image=reconstructed_image.astype(np.uint8)
# print(reconstructed_image.dtype)
#
# #Now save it as segmented volume.
# from tifffile import imwrite
# imwrite('results/vnet.tif', reconstructed_image)


