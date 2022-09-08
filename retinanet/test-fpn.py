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


classification_head = PredictionHead(20*9, num_conv_layers=0, bias_initializer=prior_probability)
box_head = PredictionHead(4*9, num_conv_layers=0, bias_initializer='zeros')


# In[10]:


model = keras_cv.models.RetinaNet(
    classes=20,
    feature_pyramid = FeaturePyramid(),
    classification_head=classification_head,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
    evaluate_train_time_metrics=False,
)
# Fine-tuning a RetinaNet is as simple as setting backbone.trainable = False
model.backbone.trainable = True


# In[ ]:


optimizer = tf.optimizers.SGD(learning_rate=0.1, momentum=0.9, global_clipnorm=10.0)
model.compile(
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    optimizer=optimizer,
)


callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs"),
    keras.callbacks.EarlyStopping(patience=15),
    keras.callbacks.ReduceLROnPlateau(patience=10),
    # Uncomment to train your own RetinaNet
    keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True),
]

model.fit(
    train_ds,
    validation_data=val_ds.take(20),
    epochs=EPOCHS,
    callbacks=callbacks,
)


# In[ ]:


model.save_weights(CHECKPOINT_PATH)


# In[ ]:


"""
An important nuance to note is that by default the KerasCV RetinaNet does not evaluate
metrics at train time.  This is to ensure optimal GPU performance and TPU compatibility.
If you want to evaluate train time metrics, you may pass
`evaluate_train_time_metrics=True` to the `keras_cv.models.RetinaNet` constructor.
"""

"""
## Evaluation with COCO Metrics

KerasCV offers a suite of in-graph COCO metrics that support batch-wise evaluation.
More information on these metrics is available in:

- [Efficient Graph-Friendly COCO Metric Computation for Train-Time Model Evaluation](https://arxiv.org/abs/2207.12120)
- [Using KerasCV COCO Metrics](https://keras.io/guides/keras_cv/coco_metrics/)

Let's construct two COCO metrics, an instance of
`keras_cv.metrics.COCOMeanAveragePrecision` with the parameterization to match the
standard COCO Mean Average Precision metric, and `keras_cv.metrics.COCORecall`
parameterized to match the standard COCO Recall metric.
"""

metrics = [
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
]


"""
Next, we can evaluate the metrics by re-compiling the model, and running
`model.evaluate()`:
"""

model.load_weights(INFERENCE_CHECKPOINT_PATH)
model.compile(
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    optimizer=tf.optimizers.SGD(learning_rate=0.1, momentum=0.9, global_clipnorm=10.0),
    metrics=metrics,
)
metrics = model.evaluate(val_ds.take(20), return_dict=True)
print(metrics)
# {"Mean Average Precision": 0.612, "Recall": 0.767}

"""
## Inference

KerasCV makes object detection inference simple.  `model.predict(images)` returns a
RaggedTensor of bounding boxes.  By default, `RetinaNet.predict()` will perform
a non max suppression operation for you.
"""

model = keras_cv.models.RetinaNet(
    classes=20,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
)
model.load_weights(INFERENCE_CHECKPOINT_PATH)


def visualize_detections(model):
    train_ds, val_dataset_info = keras_cv.datasets.pascal_voc.load(
        bounding_box_format="xywh", split="train", batch_size=9
    )
    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    images, labels = next(iter(train_ds.take(1)))
    predictions = model.predict(images)
    color = tf.constant(((255.0, 0, 0),))
    plt.figure(figsize=(10, 10))
    predictions = keras_cv.bounding_box.convert_format(
        predictions, source="xywh", target="rel_yxyx", images=images
    )
    predictions = predictions.to_tensor(default_value=-1)
    plotted_images = tf.image.draw_bounding_boxes(images, predictions[..., :4], color)
    for i in range(9):
        plt.subplot(9 // 3, 9 // 3, i + 1)
        plt.imshow(plotted_images[i].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


visualize_detections(model)

"""
To get good results, you should train for at least 100 epochs.  You also need to
tune the prediction decoder layer.  This can be done by passing a custom prediction
decoder to the RetinaNet constructor as follows:
"""

prediction_decoder = keras_cv.layers.NmsPredictionDecoder(
    bounding_box_format="xywh",
    anchor_generator=keras_cv.models.RetinaNet.default_anchor_generator(
        bounding_box_format="xywh"
    ),
    suppression_layer=keras_cv.layers.NonMaxSuppression(
        iou_threshold=0.75,
        bounding_box_format="xywh",
        classes=20,
        confidence_threshold=0.85,
    ),
)
model = keras_cv.models.RetinaNet(
    classes=20,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights=None,
    include_rescaling=True,
    prediction_decoder=prediction_decoder,
)
model.load_weights(INFERENCE_CHECKPOINT_PATH)
visualize_detections(model)

