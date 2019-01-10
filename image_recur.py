import cv2
import numpy as np

'''
IF this method is famous in the future, rename it to Kaleidoscope

'''
Epsilon = 1e-5

def _image_recur(img, width, height):
    '''
    helper function for seamless image recurring, imagine it as Kaleidoscope
    reference:
        U-Net: Convolutional Networks for Biomedical Image Segmentation

    :param img: source image, any channel numbers
    :param width: target image width
    :param height: target image height
    :return: recurring image
    '''
    shape = img.shape
    if len(shape) == 2:
        return _image_recur_single_channel(img, width, height)

    tar = np.ndarray([height, width, shape[-1]])
    for i in range(shape[-1]):
        tmp = _image_recur_single_channel(img[:,:,i], width, height)
        tar[:, :, i] = tmp

    # CAUTION:
    tar = np.uint8(tar)
    return tar


def _image_recur_single_channel(img, width, height):
    '''
    imaging recurring of a single channel
    '''
    shape = img.shape   # height, width

    patch = np.concatenate((img, np.flip(img, axis=1)), axis=1)
    patch = np.concatenate((patch, np.flip(patch, axis=0)), axis=0)

    shape_ = patch.shape

    _x_rep = width / shape[1]
    _y_rep = height / shape[0]

    _x_rep = np.rint(_x_rep / 2) + 1
    _y_rep = np.rint(_y_rep / 2) + 1

    _x_rep = np.int(_x_rep)
    _y_rep = np.int(_y_rep)

    patch_rep = np.tile(patch, (_y_rep, _x_rep))

    _weight_x = (width / shape[1] - Epsilon) // 4 * 2 + 2.5
    _weight_y = (height / shape[0] - Epsilon) // 4 * 2 + 2.5

    # extreme condition, when
    if width < shape[1]:
        _weight_x = 0.5
    if height < shape[0]:
        _weight_y = 0.5

    _x_start = _weight_x * shape[1] - width / 2
    _y_start = _weight_y * shape[0] - height / 2

    _x_start = np.int(_x_start)
    _y_start = np.int(_y_start)

    return patch_rep[_y_start:_y_start+height, _x_start:_x_start+width]


if __name__ == '__main__':
    '''
    run main function for a better illustration
    '''
    test_img = cv2.imread('label.jpg')
    test_img = cv2.resize(test_img, (200, 100))
    res_img = _image_recur(test_img, 250, 150)
    res_img_irregular = _image_recur(test_img, 789, 345)
    res_img_single_channel = _image_recur(test_img[:,:,0], 789, 456)
    cv2.imshow('origin', test_img)
    cv2.imshow('recur', res_img)
    cv2.imshow('irregular recur', res_img_irregular)
    cv2.imshow('single channel', res_img_single_channel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
