'''
data provider for custom training process
'''
import os, glob
import cv2

TRAINING_PATH = os.path.join(
    os.path.dirname(__file__),
    'Train'
)

class data_provider:
    def __init__(self, batch_num):
        self.nbatch = batch_num
        self.img_names = glob.glob(os.path.join(TRAINING_PATH, '*.jpg'))
        self.img_num = len(self.img_names)


    def _next_training_batch(self):
        '''
        Randomly pick training images and labels,
        return input tensor and label tensor

        :return:[input tensor: shape=[batch_num, x, y, channels],
                output tensor: shape=[batch_num, x, y, classes]
        '''



