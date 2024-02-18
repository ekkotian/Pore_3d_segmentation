import numpy as np
from skimage import io
from patchify import patchify, unpatchify
from models.vnet import vnet  # Import your specific model
from keras.models import load_model
from tifffile import imwrite

def preprocess_input(img):
    return img / 255.0

def predict_volume(model, large_image):
    # Break the large image (volume) into patches
    patches = patchify(large_image, (64, 64, 64), step=64)

    # Predict each 3D patch
    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            for k in range(patches.shape[2]):
                single_patch = patches[i, j, k, :, :, :]
                single_patch_3ch = np.stack((single_patch,) * 3, axis=-1)
                single_patch_3ch_input = preprocess_input(np.expand_dims(single_patch_3ch, axis=0))
                single_patch_prediction = model.predict(single_patch_3ch_input)
                single_patch_prediction_argmax = np.argmax(single_patch_prediction, axis=4)[0, :, :, :]
                predicted_patches.append(single_patch_prediction_argmax)

    # Convert list to numpy array
    predicted_patches = np.array(predicted_patches)

    # Reshape to the shape we had after patchifying
    predicted_patches_reshaped = np.reshape(predicted_patches,
                                            (patches.shape[0], patches.shape[1], patches.shape[2],
                                             patches.shape[3], patches.shape[4], patches.shape[5]))

    # Repach individual patches into the original volume shape
    reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)

    return reconstructed_image.astype(np.uint8)

if __name__ == "__main__":
    # Load your pre-trained model


    my_model = load_model('./pt/vnet.h5', compile=False)

    # Load the large image for prediction
    large_image = io.imread('./3D_dataset/test/test_3D.tif')

    # Perform prediction
    segmented_volume = predict_volume(my_model, large_image)

    # Save the segmented volume
    imwrite('results/vnet_segmented.tif', segmented_volume)
