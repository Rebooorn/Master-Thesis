'''
To get shape information from u-net prediction
'''

import numpy as np
import cv2
from img_provider import simple_padding_data_provider

def object_detect(img, obj):
    '''
    This widget is used to sketch the breast contour from u-net prediction

    :param img: predicted detector area from u-net(binary)
    :param obj: predicted breast area from u-net(binary)
    :return: binary mask of breast on the detector
    '''
    img = np.uint8(img)

    _, contour_locs, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)
    # for pt in contour_locs[0]:
    #     # x = np.clip(pt[0,0], 0, mask.shape[0])
    #     # y = np.clip(pt[0,1], 0, mask.shape[1])
    #     mask[pt[0,1], pt[0,0]] = 1

    hull = cv2.convexHull(contour_locs[0])
    mask = cv2.drawContours(mask, [hull], 0, 1, cv2.FILLED)

    mask = mask * obj
    return mask

if __name__ == '__main__':
    '''
    a quick demo for testing
    '''
    img = cv2.imread('D:\\ChangLiu\\MasterThesis\\Master-Thesis\\TrainSet\\Label_Class_1\\00039.jpg', 0)
    obj = cv2.imread('D:\\ChangLiu\\MasterThesis\\Master-Thesis\\TrainSet\\Label_Class_2\\00039.jpg', 0)

    window_width = 800
    window_height = 400
    # transform image to binary
    img = np.rint(img / 255.0)
    obj = np.rint(obj / 255.0)

    cv2.imshow('ori', np.float32(cv2.resize(img, (window_width, window_height))))
    cv2.imshow('obj', np.float32(cv2.resize(obj, (window_width, window_height))))

    _obj = object_detect(img, obj)

    cv2.imshow('contour', np.float32(cv2.resize(_obj, (window_width, window_height), cv2.INTER_NEAREST)))
    # cv2.imshow('contour', np.float32(_obj))

    cv2.waitKey()
    cv2.destroyAllWindows()
