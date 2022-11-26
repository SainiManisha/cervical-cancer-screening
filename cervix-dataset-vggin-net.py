#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Concatenate,
    BatchNormalization,
)
from tensorflow.keras.layers import (
    Dropout,
    Dense,
    Flatten,
)

from sklearn.metrics import classification_report
from imblearn.metrics import classification_report_imbalanced

from image_augmentation.image import RandAugment

import sys
sys.path.append("/mnt/persistent-disk/manisha_saini_gav/multiclass-imbalanced-classification")

from multi_class_imbalanced import cervix_dataset

# In[2]:


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
            devices=tf.config.list_logical_devices(),
        )

    def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
        return {"TPU": 8}


# In[3]:


resolver = LocalTPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

# In[7]:


def vgginnet_builder(image_size):
    base_model = VGG16(include_top=False, input_shape=(image_size, image_size, 3))

    layer_name = "block4_pool"
    feature_ex_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(layer_name).output,
        name="vgg16_features",
    )
    feature_ex_model.trainable = True

    def naive_inception_module(layer_in, f1, f2, f3):
        # 1x1 conv
        conv1 = Conv2D(f1, (1, 1), padding="same", activation="relu")(layer_in)
        # 3x3 conv
        conv3 = Conv2D(f2, (3, 3), padding="same", activation="relu")(layer_in)
        # 5x5 conv
        conv5 = Conv2D(f3, (5, 5), padding="same", activation="relu")(layer_in)
        # 3x3 max pooling
        pool = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(layer_in)
        # concatenate filters, assumes filters/channels last
        layer_out = Concatenate()([conv1, conv3, conv5, pool])
        return layer_out

    out = naive_inception_module(feature_ex_model.output, 64, 128, 32)
    num_classes = 3

    bn1 = BatchNormalization(name="BN")(out)
    f = Flatten()(bn1)
    dropout = Dropout(0.4, name="Dropout")(f)
    desne = Dense(num_classes, 
        activation="softmax", 
        name="Predictions")(dropout)

    model = Model(inputs=feature_ex_model.input, outputs=desne)
    return model


# In[8]:

IMAGE_SIZE = (299, 224)
CROP_SIZE = 224
BATCH_SIZE = 512
NUM_CLASSES = 3

# In[11]:

directory = (
    "/mnt/persistent-disk/manisha_saini_gav/cervix/imccs-dataset-customized")
augmenter = RandAugment(8, 2)
ds = cervix_dataset.load_image_dataset(
    directory,
    image_size=IMAGE_SIZE,
    crop_size=CROP_SIZE,
    batch_size=BATCH_SIZE,
    num_classes=NUM_CLASSES,
    rejection_resample=True,
    preprocess_fn=vgg_preprocess,
    augment_fn=augmenter.apply_on_image,
    use_tpu=True,
    use_cache=True)

dtrain, dtest = ds['train'], ds['test']

# In[13]:

with strategy.scope():
    vgginet = vgginnet_builder(CROP_SIZE)
    vgginet.summary()


# In[14]:

train_steps_per_epoch = 50
# because
# ((7625 * dtrain.minority_fraction) / 32) * 3


# In[17]:


with strategy.scope():
    vgginet.compile(
        tf.keras.optimizers.SGD(0.0001, momentum=0.9),
        # tf.keras.optimizers.Adam(0.0001),
        tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        steps_per_execution=train_steps_per_epoch,
    )

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("./cervix/vgginnet-1/model"),
    tf.keras.callbacks.TensorBoard("./cervix/vgginnet-1/tensorboard"),
]


# In[ ]:

vgginet.fit(dtrain, steps_per_epoch=train_steps_per_epoch, epochs=300, validation_data=dtest, validation_freq=10, callbacks=callbacks)

# In[ ]:

y_true, y_pred = [], []

for x, y in dtest:
    preds = vgginet.predict(x)
    y_pred.append(preds)
    y_true.append(y)

y_pred = tf.argmax(tf.concat(y_pred, 0), -1)
y_true = tf.concat(y_true, 0)

print(classification_report(y_true, y_pred))
print(classification_report_imbalanced(y_true, y_pred))
