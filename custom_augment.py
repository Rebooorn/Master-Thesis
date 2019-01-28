'''
custom image augmentation method, that is more realistic

'''

import numpy as np
from numpy import round
import cv2, os, random
import glob
import scipy.ndimage as scipyimg
from scipy.ndimage import zoom
from tf_unet.image_util import BaseDataProvider

# uncomment if no reproduction is needed
# np.random.seed(98765)



def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


class custom_augmenter:
    def __init__(self, n_class, bg_path, obj_path, **kwargs):
        self.n_class = n_class
        self.bg_path = bg_path
        self.obj_path = obj_path
        self.bg_imgs = glob.glob(bg_path+'\\ori\\*.jpg')
        self.obj_imgs = glob.glob(obj_path+'\\ori\\*.jpg')

        self.rename_imgs = kwargs.get("rename_images", False)
        if self.rename_imgs is True:
            self._rename_images()
            self.bg_imgs = glob.glob(bg_path + '\\ori\\*.jpg')
            self.obj_imgs = glob.glob(obj_path + '\\ori\\*.jpg')

        self.disp_gen = kwargs.get("disp_gen", False)

        # random parameters
        self.offset_ratio = kwargs.get('offset_ratio', 0.1)
        self.rotate_max = kwargs.get('rot_max', 10)
        self.zoom_ratio = kwargs.get('zoom_max', 0.3)

        # create saving directory if needed
        self.ori_save_path = None
        self.labels_save_path = None

    def _rename_images(self):
        '''
        rename images in bg_path and obj_path accordingly
        '''
        n_imgs = len(self.bg_imgs)

        for i, img_name in enumerate(self.bg_imgs):
            img_name = os.path.basename(img_name)
            os.rename(self.bg_path+'\\ori\\'+img_name, self.bg_path+'\\ori\\'+'{:0>5d}.jpg'.format(i))
            os.rename(self.bg_path+'\\label_cls_2\\'+img_name, self.bg_path+'\\label_cls_2\\'+'{:0>5d}.jpg'.format(i))
            os.rename(self.bg_path+'\\label_cls_3\\'+img_name, self.bg_path+'\\label_cls_3\\'+'{:0>5d}.jpg'.format(i))

        for i, img_name in enumerate(self.obj_imgs):
            img_name = os.path.basename(img_name)
            os.rename(self.obj_path+'\\ori\\'+img_name, self.obj_path+'\\ori\\'+'{:0>5d}.jpg'.format(i))
            os.rename(self.obj_path+'\\label_cls_1\\'+img_name, self.obj_path+'\\label_cls_2\\'+'{:0>5d}.jpg'.format(i))


    def sample(self, n_aug):
        '''
        randomly generate n_aug augmented images
        :param n_aug:  number of augmentations
        :return:
        '''
        for i in range(n_aug):
            ori, labels = self.generate()
            cv2.imwrite(self.ori_save_path + '{:0>5d}.jpg'.format(i), ori)
            for c in range(self.n_class):
                cv2.imwrite(self.labels_save_path[c] + '{:0>5d}.jpg'.format(i), labels[...,c])

    def generate(self):
        bg_ori_path = random.choice(self.bg_imgs)
        obj_ori_path = random.choice(self.obj_imgs)
        bg_ori = cv2.imread(bg_ori_path)
        obj_ori = cv2.imread(obj_ori_path)
        obj_label = cv2.imread(
            os.path.dirname(os.path.dirname(obj_ori_path)) + '\\label_1\\' + os.path.basename(obj_ori_path), 0
        )
        bg_label_2 = cv2.imread(
            os.path.dirname(os.path.dirname(bg_ori_path)) + '\\label_2\\' + os.path.basename(bg_ori_path), 0
        )
        bg_label_3 = cv2.imread(
            os.path.dirname(os.path.dirname(bg_ori_path)) + '\\label_3\\' + os.path.basename(bg_ori_path), 0
        )

        # Ensure label image are binary(0, 1)
        obj_label = self._from_img_to_label(obj_label)
        bg_label_2 = self._from_img_to_label(bg_label_2)
        bg_label_3 = self._from_img_to_label(bg_label_3)

        w = bg_ori.shape[0]
        h = bg_ori.shape[1]

        # random parameters
        rand_offset = [np.random.rand(1) * w * self.offset_ratio, np.random.rand(1) * h * self.offset_ratio]
        rand_offset = np.array(rand_offset)
        rand_rot = (np.random.rand(1) * 2.0 - 1.0) * self.rotate_max * np.pi / 180.0
        rand_zoom = (np.random.rand(1) * 2.0 - 1.0) * self.zoom_ratio + 1.0

        # only apply label_1 and obj, which should be breast
        # here modified according to https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab

        label_1 = obj_label
        label_tmp = cv2_clipped_zoom(label_1, rand_zoom)
        cen = 0.5 * np.array(label_1.shape)
        transform = np.array([[np.cos(rand_rot), -np.sin(rand_rot)],
                              [np.sin(rand_rot), np.cos(rand_rot)]])
        offset = cen - cen.dot(transform) + rand_offset
        # affine transform to rotate and translation label img

        # label_tmp = scipyimg.affine_transform(
        #     label_tmp, matrix=transform.T, order=2, offset=offset, output_shape=(w, h), cval=0, output=np.float32
        # )

        transform = np.array([[np.cos(rand_rot), -np.sin(rand_rot), rand_offset[0]],
                              [np.sin(rand_rot), np.cos(rand_rot), rand_offset[1]]], dtype=np.float32)
        label_tmp = cv2.warpAffine(label_tmp, transform, (h, w))

        # transform obj img accordingly
        obj_ori_tmp = cv2_clipped_zoom(obj_ori, rand_zoom)
        # obj_ori_tmp = scipyimg.affine_transform(
        #     obj_ori_tmp, matrix=transform.T, order=2, offset=offset, output_shape=(w, h), cval=0, output=np.float32
        # )
        obj_ori_tmp = cv2.warpAffine(obj_ori_tmp, transform, (h, w))

        # TODO: do the cover things
        labels = np.zeros([w, h, 4], dtype=np.float32)
        labels[...,2] = bg_label_2 * (1.0 - label_tmp)
        labels[...,1] = label_tmp * (1.0 - bg_label_3)
        labels[...,3] = bg_label_3
        labels[...,0] = 1.0 - np.float32(np.any(labels[:, :, 1:], axis=2))

        obj_mask = labels[...,1]
        ori_mask = np.dstack((obj_mask, obj_mask, obj_mask))
        ori_combined = bg_ori * (1.0 - ori_mask) + obj_ori_tmp * ori_mask

        return ori_combined, labels

    def _from_img_to_label(self, img):
        '''transform image from 0-255 to binary'''
        return np.rint(img/255.0)


class custom_data_provider(BaseDataProvider):
    def __init__(self, n_class, bg_path, obj_path, **kwargs):
        super().__init__()
        self.n_class = n_class
        self.bg_path = bg_path
        self.obj_path = obj_path
        self.generator = custom_augmenter(n_class=self.n_class,
                                          bg_path=self.bg_path,
                                          obj_path=self.obj_path)




if __name__ == '__main__':
    tester = custom_augmenter(n_class=3,
                              bg_path='D:\\ChangLiu\\MasterThesis\\custom-augmentation\\background',
                              obj_path='D:\ChangLiu\MasterThesis\custom-augmentation\\object',
                              )
    w = int(1936 / 2)
    h = int(1096 / 2)
    for _ in range(100):
        src, label = tester.generate()
        src = cv2.resize(src, (w, h))
        label = cv2.resize(label, (w, h))
        label = np.argmax(label, axis=-1) / 3.0
        cv2.imshow('src', src / 255.0)
        cv2.imshow('label', np.float32(label))
        k = cv2.waitKey()
        if k == ord('q'):
            break

    cv2.destroyAllWindows()