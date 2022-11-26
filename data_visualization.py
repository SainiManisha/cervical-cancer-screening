# In[]:
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append("/mnt/persistent-disk/manisha_saini_gav/multiclass-imbalanced-classification")
from multi_class_imbalanced import cervix_dataset

IMAGE_SIZE = (299, 224)
CROP_SIZE = 224
BATCH_SIZE = 512
NUM_CLASSES = 3
EPOCHS = 300
IMAGES_DIRECTORY = "/mnt/persistent-disk/manisha_saini_gav/cervix/imccs-dataset-customized"

# In[]:
ds = cervix_dataset.load_image_dataset(
    IMAGES_DIRECTORY,
    image_size=IMAGE_SIZE,
    crop_size=CROP_SIZE,
    batch_size=BATCH_SIZE,
    num_classes=NUM_CLASSES,
    rejection_resample=True,
    preprocess_fn=None,
    augment_fn=None,
    use_tpu=True,
    use_cache=True)

dtrain, dtest = ds['train'], ds['test']
train_distro = ds['train_distro']

for x, y in dtrain:
    break

# In[]:
labels = ['Type 1', 'Type 2', 'Type 3']
train_distro = train_distro.numpy()

plt.style.use('ggplot')
plt.bar(labels, train_distro, color=np.array(
    [[0, 0.5, 0], [0.7, 0.3, 0.0], [0.6, 0.3, 0.3]]
))
plt.title("Distribution of Samples per Class")
plt.savefig("cervix-histogram-figure2.svg")
plt.show()

# In[]:

x, y = tf.random.shuffle(x), tf.random.shuffle(y)
images, labels = x[:25], y[:25]

plt.figure(figsize=[13, 13])
for i, (image, label) in enumerate(zip(images, labels)):
    plt.subplot(5, 5, i + 1)
    plt.imshow(image / 255.)
    plt.axis("off")

    label = f'Type {int(label) + 1}'
    plt.title(label)
plt.savefig("cervix-dataset-figure1.svg")
plt.show()
