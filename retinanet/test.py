import tensorflow as tf

from tensorflow.keras import optimizers
import keras_cv
from keras_cv.models.object_detection.retina_net.__internal__ import (
    layers as layers_lib,
)

def _create_bounding_box_dataset(bounding_box_format):

    # Just about the easiest dataset you can have, all classes are 0, all boxes are
    # exactly the same.  [1, 1, 2, 2] are the coordinates in xyxy
    xs = tf.ones((10, 512, 512, 3), dtype=tf.float32)
    y_classes = tf.zeros((10, 10, 1), dtype=tf.float32)

    ys = tf.constant([0.25, 0.25, 0.1, 0.1], dtype=tf.float32)
    ys = tf.expand_dims(ys, axis=0)
    ys = tf.expand_dims(ys, axis=0)
    ys = tf.tile(ys, [10, 10, 1])
    ys = tf.concat([ys, y_classes], axis=-1)

    ys = keras_cv.bounding_box.convert_format(
        ys, source="rel_xywh", target=bounding_box_format, images=xs, dtype=tf.float32
    )
    return xs, ys

bounding_box_format = "xywh"
retina_net = keras_cv.models.RetinaNet(
    classes=1,
    bounding_box_format=bounding_box_format,
    backbone="resnet50",
    backbone_weights=None,
    include_rescaling=False,
    evaluate_train_time_metrics=True,
)

retina_net.compile(
    optimizer=optimizers.Adam(),
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    metrics=[
        keras_cv.metrics.COCOMeanAveragePrecision(
            class_ids=range(1),
            bounding_box_format=bounding_box_format,
            name="MaP",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=range(1),
            bounding_box_format=bounding_box_format,
            name="Recall",
        ),
    ],
)

x, y = _create_bounding_box_dataset('xywh')
retina_net.fit(x, y)
