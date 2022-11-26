import glob
import os
import random
from matplotlib import pyplot as plt

base_ds_path = '/home/manisha_saini_gav/imccs-dataset-customized/additional'
subdirectories = ['Type_1', 'Type_2', 'Type_3']

for subdir in subdirectories:
    image_paths = glob.glob(os.path.join(base_ds_path, subdir, '*.jpg'))
    random.shuffle(image_paths)
    images = [
        plt.imread(image_path) for image_path in image_paths[:6]
    ]

    plt.figure(figsize=[10, 15])
    for i, image in enumerate(images):
        plt.subplot(3, 2, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.suptitle(subdir)
    plt.tight_layout()
    plt.savefig(subdir + '.svg')
