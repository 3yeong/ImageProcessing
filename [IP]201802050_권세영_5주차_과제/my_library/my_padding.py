import cv2
import numpy as np

def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        print('repetition padding')

        pad_img[: p_h, p_w:p_w + w] = src[0, :]

        pad_img[p_h + h:, p_w:p_w + w] = src[h - 1, :]

        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]

        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1: p_w + w]

    else:
        print('zero padding')

    return pad_img
