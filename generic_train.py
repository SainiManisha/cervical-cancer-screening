#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

from sklearn.metrics import classification_report
from imblearn.metrics import classification_report_imbalanced

from image_augmentation.image import RandAugment

import sys
sys.path.append("/home/manisha_saini_gav/multiclass-imbalanced-classification")
from multi_class_imbalanced import cervix_dataset


IMAGE_SIZE = (299, 224)
CROP_SIZE = 224
BATCH_SIZE = 512
NUM_CLASSES = 3
EPOCHS = 300
IMAGES_DIRECTORY = "/home/manisha_saini_gav/imccs-dataset-customized"


class LocalTPUClusterResolver(tf.distribute.cluster_resolver.TPUClusterResolver):
    """LocalTPUClusterResolver, should be compatible with TPU v3-8 and v2-8."""

    def __init__(self):
        self._tpu = ""
        self.task_type = "worker"
        self.task_id = 0

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        return None

    def cluster_spec(self):
        return tf.train.ClusterSpec({})

    def get_tpu_system_metadata(self):
        return tf.tpu.experimental.TPUSystemMetadata(
            num_cores=8,
            num_hosts=1,
            num_of_cores_per_host=8,
            topology=None,
            devices=tf.config.list_logical_devices())

    def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
        return {"TPU": 8}


def train_on_cervix(create_model_fn, model_dir, preprocess_fn, augment=True, reject_resample=True, optimizer_fn=None):
    """Trains a tf.keras.Model on the Intel Mobile-ODT Cervical Cancer Screening Dataset and store evaluation
    metrics to "<model_dir>/results.txt" file. During training, callbacks perform periodic model.save() to
    "<model_dir>/saved_model" and tensorboard logs to "<model_dir>/tensorboard".

    Args:
        create_model_fn: function to create an instance of tf.keras.Model that will be used to train the model.
            Effectively, this function is invoked inside TPUStrategy to ensure that model is created and
            trained on TPU.
        model_dir: 
        preprocess_fn: function to be used for pre-processing the images for all train and test samples.
        augment: boolean whether to apply data augmentation on the training pipeline or not.
            If true (by default), RandAugment(m=8, n=2) is applied.
        reject_resample: boolean whether to apply rejection resampling technique on the training pipeline or not.
            If true (by default), random undersampling of data samples are applied during training which aims to
            achieve class balanced distribution of training samples continually.
        optimizer_fn: function to create an instance of tf.keras.optimizer.Optimizer that will be used for training.
            If unspecified (set to None), tf.keras.SGD(learning_rate=0.0001, momentum=0.9).
    
    Returns:
        model: trained model
    """

    resolver = LocalTPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    if augment:
        augmenter = RandAugment(8, 2)
        augment_fn = augmenter.apply_on_image
    else:
        augment_fn = None

    ds = cervix_dataset.load_image_dataset(
        IMAGES_DIRECTORY,
        image_size=IMAGE_SIZE,
        crop_size=CROP_SIZE,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        rejection_resample=reject_resample,
        preprocess_fn=preprocess_fn,
        augment_fn=augment_fn,
        use_tpu=True,
        use_cache=True)

    dtrain, dtest = ds["train"], ds["test"]

    with strategy.scope():
        model = create_model_fn()
        model.summary()

    train_steps_per_epoch = 50
    # because ((7625 * dtrain.minority_fraction) / 32) * 3
    
    if optimizer_fn == None:
        optimizer_fn = lambda: tf.keras.optimizers.SGD(0.0001, momentum=0.9)

    with strategy.scope():
        model.compile(
            optimizer_fn(),
            tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            steps_per_execution=train_steps_per_epoch)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_dir + "/saved_model"),
        tf.keras.callbacks.TensorBoard(model_dir + "/tensorboard")]

    model.fit(
        dtrain,
        steps_per_epoch=train_steps_per_epoch,
        epochs=EPOCHS,
        validation_data=dtest,
        validation_freq=10,
        callbacks=callbacks)

    y_true, y_pred = [], []

    for x, y in dtest:
        preds = model.predict(x)
        y_pred.append(preds)
        y_true.append(y)

    y_pred = tf.argmax(tf.concat(y_pred, 0), -1)
    y_true = tf.concat(y_true, 0)

    cr1 = classification_report(y_true, y_pred)
    cr2 = classification_report_imbalanced(y_true, y_pred)

    with open(model_dir + "/results.txt", "w") as results_file:
        print(cr1, file=results_file)
        print(cr1)
        print(cr2, file=results_file)
        print(cr2)

    return model
