# Computer-Vision
Zombie Detection Model Training
Welcome to the Zombie Detection model training assignment! This guide will walk you through the steps to train a RetinaNet model to detect zombies using the TensorFlow Object Detection API. Follow the steps below to complete the assignment.

Table of Contents:
1. Installation
2. Import Necessary Packages
3. Visualize Training Images
4. Define the Category Index Dictionary
5. Download and Configure Checkpoints
6. Modify Model Configuration
7. Build and Fine-tune the Model
8. Run a Dummy Image
9. Set Training Hyperparameters
10. Select Prediction Layer Variables
11. Define the Training Step
12. Run the Training Loop
13. Test the Model
14. Preprocess, Predict, and Post-process
15. Save Results for Grading


1. Installation

First, clone the TensorFlow Model Garden and install the TensorFlow Object Detection API.

!rm -rf ./models/
!git clone --depth 1 https://github.com/tensorflow/models/
!sed -i 's/tf-models-official>=2.5.1/tf-models-official==2.15.0/g' ./models/research/object_detection/packages/tf2/setup.py
!cd models/research/ && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .

2. Import Necessary Packages
Import the necessary modules for object detection.

from object_detection.utils import label_map_util, config_util, visualization_utils as viz_utils, colab_utils
from object_detection.builders import model_builder
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import zipfile
from PIL import Image

3. Visualize Training Images
Load and visualize the training images.

train_image_dir = './training'
train_images_np = []

for i in range(1, 6):
    image_path = os.path.join(train_image_dir, 'training-zombie' + str(i) + '.jpg')
    train_images_np.append(load_image_into_numpy_array(image_path))

plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labelsize'] = False
plt.rcParams['ytick.labelsize'] = False
plt.rcParams['figure.figsize'] = [14, 7]

for idx, train_image_np in enumerate(train_images_np):
    plt.subplot(1, 5, idx+1)
    plt.imshow(train_image_np)
plt.show()

4. Define the Category Index Dictionary
Define the category index dictionary for the zombie class.

zombie_class_id = 1
category_index = {
    zombie_class_id: {
        'id': zombie_class_id,
        'name': 'zombie'
    }
}
num_classes = 1

5. Download and Configure Checkpoints
Download the RetinaNet checkpoint and configure it.

!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
!tar -xf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
!mv ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint models/research/object_detection/test_data/

