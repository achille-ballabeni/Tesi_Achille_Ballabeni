import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import cv2
import pandas as pd

import sys
sys.path.append('/home/usr/local/lib/python3.7/dist-packages')

# Define paths
cov_mat_path = 'path_to_covariance_matrix/covariance_matrix.npy'
std_vec_path = 'path_to_std_vector/std_vector.npy'
dataset = 'path_to_dataset_directory/dataset/'
models = 'path_to_TF_lite_models_directory/TF_lite_models/'
ground_truth = 'path_to_ground_truth_directory/ground_truth/'

# Create lists to load all test images and models
num_images=[346,346, 346, 347,347]
img_dir=[[],[],[],[],[]]
models_dir=[]
gt_dir=[]
i=-1
for test_dir in sorted(os.listdir(dataset)):
    i+=1
    test_dir = os.path.join(dataset, test_dir)
    print(test_dir)

    for k in range(num_images[i]):
        full_path = test_dir + '/' + str(k) + '.npy'
        img_dir[i].append(full_path)

for path in sorted(os.listdir(models)):
    full_path = os.path.join(models, path)
    models_dir.append(full_path)
    print(full_path)

for path in sorted(os.listdir(ground_truth)):
    full_path = os.path.join(ground_truth, path)
    gt_dir.append(full_path)
    print(full_path)

# Preprocessing values
max_reflectance_PCA = 6.282844
max_gt_values = [325, 625, 400, 14]
mse_baseline = [8.7002814e+02, 3.8284080e+03, 1.5888525e+03, 6.7716144e-02]
n_pca_bands = 3
target_image_size = 32

cov = np.load(cov_mat_path)
std_vec = np.load(std_vec_path)


# Preprocessing functions
def reshape_and_normalize(image, std_vec, height, width):
    
    data = np.reshape(image, [height*width, 150])
    data = np.transpose(data)

    data = data/std_vec

    return data

def eig_decomposition(cov, n_pca_bands):

    [eig_values, eig_vectors] = np.linalg.eigh(cov, 'U')
    eig_vectors = eig_vectors[:, 150-n_pca_bands:]

    return eig_vectors

def apply_pca(data, n_pca_bands, eig_vectors, height, width):

    z = np.matmul(eig_vectors.T, data)
    z = np.transpose(z)
    image = np.reshape(z, [height, width, n_pca_bands])

    return image

def image_repetition_test(image, height, width):    
    nx = np.floor_divide(target_image_size, width)
    ny = np.floor_divide(target_image_size, height)
    
    image = np.tile(image, [ny, nx, 1])

    if np.maximum(nx, ny)==1:
        #image = tf.image.resize(image, [target_image_size, target_image_size], method='bilinear', antialias=False)
        image = cv2.resize(image, [target_image_size, target_image_size], interpolation=cv2.INTER_LINEAR)
    else:
        [height, width] = np.shape(image)[:2]
        image = np.pad(image, ((0,target_image_size-height),(0,target_image_size-width),(0,0)), 'constant', constant_values=0)
    return image
    
def pad_with_patches_test(image, height, width):
    max_dim = np.maximum(height, width)
    
    if max_dim < target_image_size:
        image = image_repetition_test(image, height, width)
    else:
        image = cv2.resize(image, [target_image_size, target_image_size], interpolation=cv2.INTER_LINEAR)

    return image

def post_pca_normalization(image, max_reflectance_PCA):

    image = image/max_reflectance_PCA

    return image

# Eigen decoposition of covariance matrix
eig_vectors = eig_decomposition(cov, n_pca_bands)

# Initialize variables to export data
predictions = [np.zeros((num_test_images, 4)) for num_test_images in num_images]
image_loading_time = [np.zeros((num_test_images, 4)) for num_test_images in num_images]
image_overall_time = [np.zeros((num_test_images, 4)) for num_test_images in num_images]
network_inference_time = [np.zeros((num_test_images, 4)) for num_test_images in num_images]
score = np.zeros((1,5))

test_id=-1
for test_data, model in zip(img_dir, models_dir):
    test_id += 1
    print(test_id)
    i=-1

    # Load model
    loaded_model = tf.keras.models.load_model(model, custom_objects=None, compile=False, options=None)

    for image_path in test_data:
        i += 1
        image_time_start = time.time()  # Time start

        image = np.load(image_path)

        image_loading_time[test_id][i,0] = time.time() - image_time_start   # Loading time

        h = np.shape(image)[0]
        w = np.shape(image)[1]

        # Preprocessing
        image = reshape_and_normalize(image, std_vec, h, w)
        image = apply_pca(image, n_pca_bands, eig_vectors, h, w)
        image = pad_with_patches_test(image, h, w)
        image = post_pca_normalization(image, max_reflectance_PCA)
        image = np.expand_dims(image, 0)
        image = np.float32(image)

        # Predictions
        inferece_time_start = time.time()   # Inference time start

        pred = loaded_model.predict(image, verbose=0)
        
        network_inference_time[test_id][i,0] = time.time() - inferece_time_start    # Inference time

        predictions[test_id][i,:] = pred    

        image_overall_time[test_id][i,0] = time.time() - image_time_start   # Overall time

# Define custom metric for evaluation
def custom_metric(y_true, y_pred, max_gt_values):

    y_pred = np.multiply(y_pred, max_gt_values)
    mse = np.mean((y_true-y_pred)**2, axis=0)
    score = np.mean(mse/mse_baseline)

    return score

# Export info
base_path = 'path_to_expor_directory/TF_results/'

i=-1
for pred, gt_path in zip(predictions, gt_dir):
    i += 1
    gt = np.array(pd.read_csv(gt_path))[:,1:]
    score[0,i] = custom_metric(gt, pred, max_gt_values)

print(score)

score_df = pd.DataFrame(score.T, columns=['score'])
score_df.to_csv(os.path.join(base_path, 'score.csv'), index_label='Test set index')

i=0
for ilt, iot, nit in zip(image_loading_time, image_overall_time, network_inference_time):
    i+=1
    times = pd.DataFrame(np.concatenate((ilt, nit, iot), axis = 1), columns=['image loading time','network inference time','image overall time'])
    times.to_csv(os.path.join(base_path, 'times_'+str(i)+'.csv'), index_label='Image index')