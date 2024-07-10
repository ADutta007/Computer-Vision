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

Let's start!

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

6. Modify Model Configuration
Locate and modify the model configuration.


pipeline_config = 'models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']

model_config.ssd.num_classes = num_classes
model_config.ssd.freeze_batchnorm = True

7. Build and Fine-tune the Model
Build and fine-tune the detection model.


detection_model = model_builder.build(model_config=model_config, is_training=True)

tmp_box_predictor_checkpoint = tf.compat.v2.train.Checkpoint(
    _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
    _box_prediction_head=detection_model._box_predictor._box_prediction_head
)

tmp_model_checkpoint = tf.compat.v2.train.Checkpoint(
    _feature_extractor=detection_model._feature_extractor,
    _box_predictor=tmp_box_predictor_checkpoint
)

checkpoint_path = 'models/research/object_detection/test_data/checkpoint/ckpt-0'
checkpoint = tf.compat.v2.train.Checkpoint(model=tmp_model_checkpoint)
checkpoint.restore(checkpoint_path).expect_partial()

8. Run a Dummy Image
Run a dummy image through the model to generate the variables.


tmp_image, tmp_shapes = detection_model.preprocess(tf.zeros((1, 640, 640, 3)))
tmp_prediction_dict = detection_model.predict(tmp_image, tmp_shapes)
tmp_detections = detection_model.postprocess(tmp_prediction_dict, tmp_shapes)

assert len(detection_model.trainable_variables) > 0, "Please pass in a dummy image to create the trainable variables."

9. Set Training Hyperparameters
Set the training hyperparameters.


tf.keras.backend.set_learning_phase(True)
batch_size = 5
num_batches = 200
learning_rate = 0.01
optimizer = tf.keras.optimizers.SGD(learning_rate, 0.9)

10. Select Prediction Layer Variables
Select the prediction layer variables to fine-tune.


to_fine_tune = []
prefixes_to_train = [
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead'
]
for var in detection_model.trainable_variables:
    if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
        to_fine_tune.append(var)

11. Define the Training Step
Define the training step function.


@tf.function
def train_step_fn(image_list, groundtruth_boxes_list, groundtruth_classes_list, model, optimizer, vars_to_fine_tune):
    model.provide_groundtruth(groundtruth_boxes_list=groundtruth_boxes_list, groundtruth_classes_list=groundtruth_classes_list)
    with tf.GradientTape() as tape:
        preprocessed_image_tensor = tf.concat([detection_model.preprocess(img)[0] for img in image_list], axis=0)
        true_shape_tensor = tf.constant([5, 640, 640, 3], dtype=tf.int32)
        prediction_dict = model.predict(preprocessed_image_tensor, true_shape_tensor)
        loss_dict = model.loss(prediction_dict, true_shape_tensor)
        total_loss = loss_dict['Loss/localization_loss'] + loss_dict['Loss/classification_loss']
        gradients = tape.gradient(total_loss, vars_to_fine_tune)
        optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
    return total_loss

12. Run the Training Loop
Run the training loop to fine-tune the model.


print('Start fine-tuning!', flush=True)
for idx in range(num_batches):
    all_keys = list(range(len(train_images_np)))
    random.shuffle(all_keys)
    example_keys = all_keys[:batch_size]
    gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
    gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
    image_tensors = [train_image_tensors[key] for key in example_keys]
    total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list, detection_model, optimizer, to_fine_tune)
    if idx % 10 == 0:
        print(f'batch {idx} of {num_batches}, loss={total_loss.numpy()}', flush=True)
print('Done fine-tuning!')

13. Test the Model
Download and load the test images.


!rm zombie-walk-frames.zip
!rm -rf ./results
!wget --no-check-certificate https://storage.googleapis.com/tensorflow-3-public/datasets/zombie-walk-frames.zip -O zombie-walk-frames.zip
local_zip = './zombie-walk-frames.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./results')
zip_ref.close()

test_image_dir = './results/'
test_images_np = []
for i in range(0, 237):
    image_path = os.path.join(test_image_dir, 'zombie-walk' + "{0:04}".format(i) + '.jpg')
    test_images_np.append(np.expand_dims(load_image_into_numpy_array(image_path), axis=0))

14. Preprocess, Predict, and Post-process
Define the detect function and run inference on the test images.


@tf.function
def detect(input_tensor):
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    detections







