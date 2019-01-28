'''
data provider for custom training process
'''
import os, glob
import cv2
import numpy as np
from image_recur import _image_recur
from tf_unet.image_util import BaseDataProvider
# import tensorflow as tf


TRAINING_PATH = os.path.join(
    os.path.dirname(__file__),
    'TrainSet'
)

TESTING_PATH = os.path.join(
    os.path.dirname(__file__),
    'TestSet'
)


class simple_data_provider(BaseDataProvider):
    def __init__(self, data_shuffle = True, nclass=2, **kwargs):
        '''
        A simpler data provider, no minibatch used here. When called:
        a = simple_data_provider(...)
        image, label = a(1)
        image.shape = [1, x, y, channels]
        label.shape = [1, x, y, 2]

        :param batch_num: number of images in a single batch, default is 1
        :param data_shuffle: True if image shuffle is needed
        :param nclass: number of segmentation classes
        :param kwargs:
            'channel':  'red'   :use R channel as input tensor
                        'grey'  :use grey-value as input (default)
                        'full'  :use full channels
            'resize_method':   (must for minibatch training)
                        'scale' :scale images to symm1etric size linearly
                        'recur' :scale images to same size using imaging recurring
            'x'     :   image width (must for minibatch)
            'y'     :   image height
            'test'  :
        '''
        super().__init__()

        if kwargs.get('test') is False:
            self.src_path = TRAINING_PATH
        else:
            self.src_path = TESTING_PATH

        self.img_names = glob.glob(os.path.join(self.src_path, 'Origin_img', '*.jpg'))
        self.img_num = len(self.img_names)
        self.data_shuffle = data_shuffle
        self.n_class = nclass
        # input tensor size
        if kwargs.get('x') is not None and kwargs.get('y') is not None:
            self.x = kwargs.get('x')    # width
            self.y = kwargs.get('y')    # height
        else:
            self.x = None
            self.y = None

        if kwargs.get('channel') == 'red':
            # opencv order is BGR
            self.channels = 1
        elif kwargs.get('channel') == 'full':
            self.channels = 3
        else:
            # use grey-value
            self.channels = 1
        self.kwargs = kwargs


    def _next_data(self):
        '''
        Randomly pick training images and labels,
        return input tensor and label tensor

        :return:[input tensor: shape=[batch_num, x, y, channels],
                output tensor: shape=[batch_num, x, y, classes]
        '''
        img_index = np.random.randint(self.img_num, size=1)

        if self.data_shuffle:
            np.random.shuffle(img_index)

        # TODO: how to deal with .tiff file
        img_name = self._get_image_names(img_index)
        # TODO: How to train the network with different tensor sizes

        # not using mini-batch method
        img_orig = cv2.imread(os.path.join(self.src_path, 'Origin_img', img_name[0]))

        # determine channels in input tensor
        if self.kwargs.get('channel') == 'red':
            # opencv order is BGR
            img_orig = img_orig[:, :, -1]
            img_orig = np.reshape(img_orig, [img_orig.shape[0], img_orig.shape[1], 1])
        elif self.kwargs.get('channel') == 'full':
            img_orig = img_orig
        else:
            # use grey-value
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
            img_orig = np.reshape(img_orig, [img_orig.shape[0], img_orig.shape[1], 1])

        # a fake 1st label
        label = cv2.imread(os.path.join(self.src_path, 'Label_Class_1', img_name[0]), 0)
        label = np.reshape(label, (label.shape[0], label.shape[1], 1))
        for i in range(1, self.n_class):
            cls = cv2.imread(os.path.join(self.src_path, 'Label_Class_'+str(i), img_name[0]), 0)
            cls = np.reshape(cls, (cls.shape[0], cls.shape[1], 1))
            label = np.concatenate((label, cls), axis=2)

        label = self._from_img_to_label(label)
        # change class 0 accordingly
        label[:, :, 0] = 1 - np.clip(np.sum(label[:, :, 1:], axis=2), 0, 1)

        # resize input tensor and label tensor according to preset size
        if self.x is not None and self.y is not None:
            img_orig = cv2.resize(img_orig, (self.x, self.y), interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, (self.x, self.y), interpolation=cv2.INTER_NEAREST)

        # input_tensor = np.reshape(img_orig, (1, img_orig.shape[0], img_orig.shape[1], img_orig.shape[2]))
        # input_tensor = np.reshape(img_orig, (1, img_orig.shape[0], img_orig.shape[1], 1))
        # label_tensor = np.reshape(label, (1, label.shape[0], label.shape[1], label.shape[2]))
        input_tensor = img_orig
        label_tensor = label

        return input_tensor, label_tensor


    # Override
    def _process_labels(self, label):
        return label

    # # Override
    # def _post_process(self, data, labels):
    #     # Data augmentation


    def _get_image_names(self, img_index):
        '''Transform image index to complete image names'''
        img_names = []
        for ind in img_index:
            img_names.append('{:0>5d}.jpg'.format(ind))
        return img_names

    def _img_resize(self, img):
        ''''''
        if self.kwargs.get('resize_method') == 'scale':
            res = cv2.resize(img, (self.x, self.y), interpolation=cv2.INTER_NEAREST)
            res = np.uint8(res)
        elif self.kwargs.get('resize_method') == 'recur':
            res = _image_recur(img, self.x, self.y)
        else:
            raise NameError('Resizing method is not identified')

        return res

    def _from_img_to_label(self, img):
        '''transform image from 0-255 to binary'''
        return np.rint(img/255.0)


class minibatch_data_provider(BaseDataProvider):
    '''
    TODO: THIS IS NOT 100% IMPLEMENTED
    '''
    def __init__(self, batch_num = 1, data_shuffle = True, nclass=2, **kwargs):
        '''

        :param batch_num: number of images in a single batch, default is 1
        :param data_shuffle: True if image shuffle is needed
        :param nclass: number of segmentation classes
        :param kwargs:
            'channel':  'red'   :use R channel as input tensor
                        'grey'  :use grey-value as input (default)
                        'full'  :use full channels
            'resize_method':   (must for minibatch training)
                        'scale' :scale images to symm1etric size linearly
                        'recur' :scale images to same size using imaging recurring
            'x'     :   image width (must for minibatch)
            'y'     :   image height
        '''
        super().__init__()
        self.nbatch = batch_num

        self.img_names = glob.glob(os.path.join(TRAINING_PATH, 'Origin_img', '*.jpg'))
        self.img_num = len(self.img_names)
        self.data_shuffle = data_shuffle
        self.nclass = nclass
        # input tensor size
        if kwargs.get('x') is not None and kwargs.get('y') is not None:
            self.x = kwargs.get('x')    # width
            self.y = kwargs.get('y')    # height
        else:
            self.x = None
            self.y = None

        if kwargs.get('channel') == 'red':
            # opencv order is BGR
            self.channels = 1
        elif kwargs.get('channel') == 'full':
            self.channels = 3
        else:
            # use grey-value
            self.channels = 1
        self.kwargs = kwargs

        # training using non-batch with a single image.
        # TODO: Training with minibatch with imaging resize or recur
        if self.nbatch > 1 and kwargs.get('resize_method') is None:
            raise NameError('mini-batch method needs clarifying resizeing method')

    def _next_data(self):
        '''
        Randomly pick training images and labels,
        return input tensor and label tensor

        :return:[input tensor: shape=[batch_num, x, y, channels],
                output tensor: shape=[batch_num, x, y, classes]
        '''
        img_index = np.random.randint(self.img_num, size=self.nbatch)

        if self.data_shuffle:
            np.random.shuffle(img_index)

        # TODO: how to deal with .tiff file
        img_names = self._get_image_names(img_index)
        # TODO: How to train the network with different tensor sizes

            # mini-batch method
        if self.x is None or self.y is None:
            raise NameError('For mini-batch training, sizes of input tensor are needed')
        input_tensor = np.ndarray([self.nbatch, self.y, self.x, self.channels])
        label_tensor = np.ndarray([self.nbatch, self.y, self.x, self.nclass])
        for i, filename in enumerate(img_names):
            img_orig = cv2.imread(os.path.join(TRAINING_PATH, 'Origin_img', filename))
            # determine channels in input tensor
            if self.kwargs.get('channel') == 'red':
                # opencv order is BGR
                img_orig = img_orig[:, :, -1]
            elif self.kwargs.get('channel') == 'full':
                img_orig = img_orig
            else:
                # use grey-value
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

            # resizing image to preset image size
            img_orig = self._img_resize(img_orig)
            input_tensor[i, :, :, :] = img_orig

            label = cv2.imread(os.path.join(TRAINING_PATH, 'Label_Class_1', filename), 0)
            label = np.reshape(label, (label.shape[0], label.shape[1], 1))
            # resize label to preset size
            label = self._img_resize(label)
            label = np.reshape(label, (label.shape[0], label.shape[1], 1))
            label = self._from_img_to_label(label)
            for j in range(1, self.nclass):
                cls = cv2.imread(os.path.join(TRAINING_PATH, 'Label_Class_' + str(j), filename), 0)
                cls = np.reshape(cls, (cls.shape[0], cls.shape[1], 1))
                # resize to symmetric image size
                cls = self._img_resize(cls)
                cls = np.reshape(cls, (cls.shape[0], cls.shape[1], 1))
                cls = self._from_img_to_label(cls)
                label = np.concatenate((label, cls), axis=2)

            # change class 0 accordingly
            label[:, :, 0] = 1 - np.clip(np.sum(label[:, :, 1:], axis=2), 0, 1)
            # zip to output tensor
            label_tensor[i, :, :, :] = label

            return input_tensor, label_tensor

    # Override
    def _process_data(self, data):

        return data

    # Override
    def _process_labels(self, label):
        return label

    # Override
    def _load_data_and_label(self):
        data, label = self._next_data()
        return data, label

    def _get_image_names(self, img_index):
        img_names = []
        for ind in img_index:
            img_names.append('{:0>5d}.jpg'.format(ind))
        return img_names

    def _img_resize(self, img):
        if self.kwargs.get('resize_method') == 'scale':
            res = cv2.resize(img, (self.x, self.y))
            res = np.uint8(res)
        elif self.kwargs.get('resize_method') == 'recur':
            res = _image_recur(img, self.x, self.y)
        else:
            raise NameError('Resizing method is not identified')

        return res

    def _from_img_to_label(self, img):
        '''transform image from 0-255 to binary'''
        return np.rint(img/255.0)

class simple_padding_data_provider(simple_data_provider):
    '''padding tensor using Kaleidoscope algorithm '''
    def __init__(self, data_shuffle=True, nclass=2, **kwargs):
        super(simple_padding_data_provider, self).__init__(data_shuffle=data_shuffle, nclass=nclass, **kwargs)
        self.kwargs = kwargs
        if self.kwargs.get('pad_size') is not None:
            self.pad_size = self.kwargs.get('pad_size')
        else:
            raise NameError('padding size is needed')

    # Override
    def _next_data(self):
        '''
                Randomly pick training images and labels,
                return input tensor and label tensor

                :return:[input tensor: shape=[batch_num, x, y, channels],
                        output tensor: shape=[batch_num, x, y, classes]
                '''
        img_index = np.random.randint(self.img_num, size=1)

        if self.data_shuffle:
            np.random.shuffle(img_index)

        img_name = self._get_image_names(img_index)

        # not using mini-batch method
        img_orig = cv2.imread(os.path.join(self.src_path, 'Origin_img', img_name[0]))

        # determine channels in input tensor
        if self.kwargs.get('channel') == 'red':
            # opencv order is BGR
            img_orig = img_orig[:, :, -1]
            img_orig = np.reshape(img_orig, [img_orig.shape[0], img_orig.shape[1], 1])
        elif self.kwargs.get('channel') == 'full':
            img_orig = img_orig
        else:
            # use grey-value
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
            img_orig = np.reshape(img_orig, [img_orig.shape[0], img_orig.shape[1], 1])

        # a fake 1st label
        label = cv2.imread(os.path.join(self.src_path, 'Label_Class_1', img_name[0]), 0)
        label = np.reshape(label, (label.shape[0], label.shape[1], 1))
        for i in range(1, self.n_class):
            cls = cv2.imread(os.path.join(self.src_path, 'Label_Class_' + str(i), img_name[0]), 0)
            cls = np.reshape(cls, (cls.shape[0], cls.shape[1], 1))
            label = np.concatenate((label, cls), axis=2)

        label = self._from_img_to_label(label)
        # change class 0 accordingly
        label[:, :, 0] = 1 - np.clip(np.sum(label[:, :, 1:], axis=2), 0, 1)

        # resize input tensor and label tensor according to preset size
        if self.x is not None and self.y is not None:
            img_orig = cv2.resize(img_orig, (self.x, self.y), interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, (self.x, self.y), interpolation=cv2.INTER_NEAREST)

        # padding
        img_orig = _image_recur(img_orig, self.pad_size+self.x, self.pad_size+self.y)
        label = _image_recur(label, self.pad_size+self.x, self.pad_size+self.y)

        # input_tensor = np.reshape(img_orig, (1, img_orig.shape[0], img_orig.shape[1], img_orig.shape[2]))
        # input_tensor = np.reshape(img_orig, (1, img_orig.shape[0], img_orig.shape[1], 1))
        # label_tensor = np.reshape(label, (1, label.shape[0], label.shape[1], label.shape[2]))
        input_tensor = img_orig
        label_tensor = label

        return input_tensor, label_tensor


if __name__ == '__main__':
    '''
    a quick demo to test
    only for test functionality
    '''
    # test for single batch data provider with nbatch 1
    n_cls = 3
    generator = simple_padding_data_provider(nclass=n_cls, channel='full', test=False, x=572, y=572, pad_size=88)
    # generator = simple_data_provider(nclass=n_cls, channel='full', test=False, x=572, y=572)

    for _ in range(30):
        input_tensor, label_tensor = generator(1)
        shape = input_tensor.shape
        input_tensor = np.reshape(input_tensor, [shape[1], shape[2], shape[3]])
        label_tensor = np.reshape(label_tensor, [shape[1], shape[2], n_cls])
        input_tensor = np.int16(input_tensor)
        cv2.imshow('ori', input_tensor/255)
        cv2.imshow('label_0', label_tensor[:, :, 0])
        cv2.imshow('label_1', label_tensor[:, :, 1])
        cv2.imshow('label_2', label_tensor[:, :, -1])
        cv2.waitKey(0)



    cv2.destroyAllWindows()


