#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import optimizers

import keras_cv
from keras_cv import bounding_box
import os

BATCH_SIZE = 8
EPOCHS = 1000
CHECKPOINT_PATH = "checkpoint/"


# In[2]:


def _create_bounding_box_dataset(bounding_box_format):

    # Just about the easiest dataset you can have, all classes are 0, all boxes are
    # exactly the same.  [1, 1, 2, 2] are the coordinates in xyxy
    xs = tf.random.uniform((1, 512, 512, 3), dtype=tf.float32)
    xs = tf.repeat(xs, repeats=64, axis=0)
    y_classes = tf.zeros((64, 1, 1), dtype=tf.float32)

    ys = tf.constant([0.25, 0.25, 0.1, 0.1], dtype=tf.float32)
    ys = tf.expand_dims(ys, axis=0)
    ys = tf.expand_dims(ys, axis=0)
    ys = tf.tile(ys, [64, 1, 1])
    ys = tf.concat([ys, y_classes], axis=-1)

    ys = keras_cv.bounding_box.convert_format(
        ys, source="rel_xywh", target=bounding_box_format, images=xs, dtype=tf.float32
    )
    return xs, ys


def visualize_dataset(dataset, bounding_box_format):
    color = tf.constant(((255.0, 0, 0),))
    plt.figure(figsize=(10, 10))
    for i, example in enumerate(dataset.take(9)):
        images, boxes = example["images"], example["bounding_boxes"]
        boxes = keras_cv.bounding_box.convert_format(
            boxes, source=bounding_box_format, target="rel_yxyx", images=images
        )
        boxes = boxes.to_tensor(default_value=-1)
        plotted_images = tf.image.draw_bounding_boxes(images, boxes[..., :4], color)
        plt.subplot(9 // 3, 9 // 3, i + 1)
        plt.imshow(plotted_images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


# train_ds is batched as a (images, bounding_boxes) tuple
# bounding_boxes are ragged
train_ds, train_dataset_info = keras_cv.datasets.pascal_voc.load(
    bounding_box_format="xywh", split="train", batch_size=BATCH_SIZE
)
val_ds, val_dataset_info = keras_cv.datasets.pascal_voc.load(
    bounding_box_format="xywh", split="validation", batch_size=BATCH_SIZE
)


def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# In[3]:


from keras_cv.models.object_detection.retina_net.__internal__ import (
    layers as layers_lib,
)

PredictionHead = layers_lib.PredictionHead


# In[4]:


# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow import keras


class FeaturePyramid(keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, inputs, training=False):
        c3_output, c4_output, c5_output = inputs
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)

        p3_output = self.conv_c3_3x3(p3_output) + p3_output
        p4_output = self.conv_c4_3x3(p4_output) + p4_output
        p5_output = self.conv_c5_3x3(p5_output) + p5_output

        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output


# In[6]:


import numpy as np

prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))


# In[8]:


classification_head = PredictionHead(
    20 * 9, num_conv_layers=0, bias_initializer=prior_probability
)
box_head = PredictionHead(4 * 9, num_conv_layers=0, bias_initializer="zeros")

model = keras_cv.models.RetinaNet(
    classes=20,
    # feature_pyramid=FeaturePyramid(),
    classification_head=classification_head,
    box_head=box_head,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
    evaluate_train_time_metrics=True,
)
# Disable all FPN
model.backbone.trainable = False
model.feature_pyramid.trainable = False

optimizer = tf.optimizers.SGD(global_clipnorm=10.0)
model.compile(
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    optimizer=optimizer,
    metrics=[
        keras_cv.metrics.COCOMeanAveragePrecision(
            class_ids=range(20),
            bounding_box_format="xywh",
            name="Mean Average Precision",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=range(20),
            bounding_box_format="xywh",
            max_detections=100,
            name="Recall",
        ),
    ],
)


def visualize_detections(model, x, y, path):
    predictions = model.predict(x)
    color = tf.constant(((255.0, 0, 0),))
    true_color = tf.constant(((0, 255.0, 255.0),))
    plt.figure(figsize=(10, 10))
    predictions = keras_cv.bounding_box.convert_format(
        predictions, source="xywh", target="rel_yxyx", images=x
    )
    y = keras_cv.bounding_box.convert_format(
        y, source="xywh", target="rel_yxyx", images=x
    )
    predictions = predictions.to_tensor(default_value=-1)
    if isinstance(y, tf.RaggedTensor):
        y = y.to_tensor(default_value=-1)

    plotted_images = x
    plotted_images = tf.image.draw_bounding_boxes(
        plotted_images, predictions[..., :4], color
    )
    plotted_images = tf.image.draw_bounding_boxes(
        plotted_images, y[..., :4], true_color
    )
    for i in range(9):
        plt.subplot(9 // 3, 9 // 3, i + 1)
        plt.imshow(plotted_images[i].numpy().astype("uint8"))
        plt.axis("off")
    plt.savefig(path)
    plt.close()


class VisualizeDetections(tf.keras.callbacks.Callback):
    def __init__(self, model, x, y, artifacts_dir):
        self.model = model
        self.x = x
        self.y = y
        self.artifacts_dir = artifacts_dir

    def on_epoch_end(self, epoch, logs):
        visualize_detections(
            self.model, self.x, self.y, f"{self.artifacts_dir}/{epoch}.png"
        )


x, y = _create_bounding_box_dataset("xywh")
callbacks = [
    keras.callbacks.EarlyStopping(patience=15, monitor="loss"),
    keras.callbacks.ReduceLROnPlateau(patience=10, monitor="loss"),
    # Uncomment to train your own RetinaNet
    keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True),
    VisualizeDetections(model, x, y, "artifacts/"),
]


model.fit(
    x,
    y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
)
