'''
This widget is implemented for the estimation of class imbalance ratio

Result:
average bg ratio is:  [0.32009401 0.46686125 0.21304737]

'''

import numpy as np
from img_provider import simple_data_provider


def imbalance_estimation():
    generator = simple_data_provider(x=572, y=572, nclass=3, channel='red', test=False)
    bg_ratio = np.ndarray([100, 3])
    for i in range(100):
        _, y_test = generator(1)
        cls_0 = y_test[0, :, :, 0]
        ratio_0 = np.sum(cls_0) / (572*572)
        cls_1 = y_test[0, :, :, 1]
        ratio_1 = np.sum(cls_1) / (572*572)
        cls_2 = y_test[0, :, :, 2]
        ratio_2 = np.sum(cls_2) / (572 * 572)
        bg_ratio[i, :] = [ratio_0, ratio_1, ratio_2]

        print('background ratio is ', '{:+.2f}'.format(ratio_0),
              ', {:+.2f}'.format(ratio_1), ', {:+.2f}'.format(ratio_2))

    print('average bg ratio is: ',  np.average(bg_ratio, axis=0))


if __name__ == '__main__':
    imbalance_estimation()
