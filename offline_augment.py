'''
Implementation using Augmentor package to offline augment training set,

1. change the number of labelling classes
2. choose the image path and labelling path
3.
'''

import Augmentor
import glob
from tkinter.filedialog import askdirectory
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

n_class = 2
n_augment = 5000

# ASK EVERYTIME
# TRAINING_PATH = askdirectory(title="Choose original input image")
#
# GROUND_TRUTH_PATH = []
# for i in range(n_class):
#     GROUND_TRUTH_PATH.append(askdirectory(title="Choose class " + str(i) + " label path"))

# PRESET PATH
TRAINING_PATH = 'D:\\ChangLiu\\MasterThesis\\Master-Thesis\\TrainSet\\Origin_img'
GROUND_TRUTH_PATH = [
    'D:\\ChangLiu\\MasterThesis\\Master-Thesis\\TrainSet\\Label_Class_1',
    'D:\\ChangLiu\\MasterThesis\\Master-Thesis\\TrainSet\\Label_Class_2'
]
TAR_PATH = [
    'D:\\ChangLiu\\MasterThesis\\augment\\training',
    'D:\\ChangLiu\\MasterThesis\\augment\\class1',
    'D:\\ChangLiu\\MasterThesis\\augment\\class2'
]

ground_truth_images = glob.glob(TRAINING_PATH+'/*.jpg')
label_images = []
for i in range(n_class):
    label_images.append(glob.glob(GROUND_TRUTH_PATH[i]+'\\*.jpg'))

collated_images_and_labels = list(zip(ground_truth_images,
                                      label_images[0],
                                      label_images[1]))
print(collated_images_and_labels)

images = [[np.asarray(Image.open(y)) for y in x] for x in collated_images_and_labels]

# create pipeline
p = Augmentor.DataPipeline(images)

# Augmentation methods
p.rotate(0.9, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(0.5)
p.zoom_random(0.9, percentage_area=0.8)
p.random_brightness(0.8, 0.51, 2.0)

g = p.generator(n_augment)

augmented_images = next(g)
print('augment image generated! ')

for i in range(n_augment):
    # ori_img = Image.fromarray(augmented_images[i][0])
    Image.fromarray(augmented_images[i][0]).save(TAR_PATH[0]+'\\{:0>5d}.jpg'.format(i))
    # rescale image to 0~255
    # label_1 = np.int8(np.rint(augmented_images[i][1] / 255.0) * 255.0)
    # label_2 = np.int8(np.rint(augmented_images[i][2] / 255.0) * 255.0)

    Image.fromarray(
        augmented_images[i][1]
    ).save(TAR_PATH[1] + '\\{:0>5d}.jpg'.format(i))

    Image.fromarray(
        augmented_images[i][2]
    ).save(TAR_PATH[2] + '\\{:0>5d}.jpg'.format(i))

# f, axarr = plt.subplots(3, 3, figsize=(12, 20))
# r_index = np.random.randint(0, len(augmented_images)-1)
# axarr[0,0].imshow(augmented_images[r_index][0])
# axarr[0,1].imshow(augmented_images[r_index][1], cmap="gray")
# axarr[0,2].imshow(augmented_images[r_index][2], cmap="gray")
#
# r_index = np.random.randint(0, len(augmented_images)-1)
# axarr[1,0].imshow(augmented_images[r_index][0])
# axarr[1,1].imshow(augmented_images[r_index][1], cmap="gray")
# axarr[1,2].imshow(augmented_images[r_index][2], cmap="gray")
#
# r_index = np.random.randint(0, len(augmented_images)-1)
# axarr[2,0].imshow(augmented_images[r_index][0])
# axarr[2,1].imshow(augmented_images[r_index][1], cmap="gray")
# axarr[2,2].imshow(augmented_images[r_index][2], cmap="gray")



plt.show()
