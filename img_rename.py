import os, sys
import glob
from tkinter.filedialog import askdirectory
import cv2

'''
convert wrong image names from a bad labelling tool
'''
def rename_img_test():
    src_path = askdirectory(title='Source directory')

    for i in range(65):
        os.rename(src_path + '/{:0>5d}.jpg'.format(i+1), src_path+'/{:0>5d}.jpg'.format(i))
        print('{:0>5d}.jpg'.format(i+1) + ' renamed to ' + '{:0>5d}.jpg'.format(i))

    print('all done')
