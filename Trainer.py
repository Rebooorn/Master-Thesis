from __future__ import division, print_function

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
plt.rcParams['image.cmap'] = 'gist_earth'
np.random.seed(98765)

# import unet module
from tf_unet import image_gen
from tf_unet import unet
from img_provider import simple_data_provider
from tf_unet import util

generator_artificial = image_gen.GrayScaleDataProvider(572, 572, cnt=20, rectangles=True)
generator_reallife = simple_data_provider(x=572, y=572, nclass=3, channel='red', test=False)

net_real = unet.Unet(channels=generator_reallife.channels,
                     n_class=generator_reallife.n_class,
                     layers=4,
                     features_root=16)

trainer_real = unet.Trainer(net_real, optimizer="momentum",
                       opt_kwargs=dict(momentum=0.2))
# train with reallife images
path_real = trainer_real.train(generator_reallife, "./unet_trained",
                          training_iters=32, epochs=200, display_step=8, restore=True)

generator_test_reallife = simple_data_provider(x=572, y=572, nclass=3, channel='red', test=True)
x_test, y_test = generator_test_reallife(1)
prediction = net_real.predict("./unet_trained/model.ckpt", x_test)

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 10))
ax[0,0].imshow(x_test[0, :, :, 0], aspect="auto")
pred = np.argmax(prediction, axis=3)
label_ = np.argmax(y_test, axis=3)
# segmentation res
ax[0,1].imshow(label_[0, :, :], aspect="auto")
ax[0,2].imshow(pred[0, :, :], aspect="auto")

# activation of separate class
ax[1,0].imshow(prediction[0, :, :, 0], aspect="auto")
ax[1,1].imshow(prediction[0, :, :, 1], aspect="auto")
ax[1,2].imshow(prediction[0, :, :, 2], aspect="auto")
# pred = np.int8(np.rint(pred))
ax[0,2].imshow(pred[0, :, :], aspect="auto")
# ax[2].imshow(prediction[0, :, :, 2], aspect="auto")

# ax[3].imshow(prediction[0,...,1] > 0.07, aspect="auto")
# ax[4].imshow(prediction[0,...,1] > 0.075, aspect="auto")
# ax[5].imshow(prediction[0,...,1] > 0.08, aspect="auto")
fig.tight_layout()
plt.show()