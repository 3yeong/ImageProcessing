import cv2
import numpy as np
from my_library.my_padding import my_padding

def my_filtering(src, filter, pad_type='zero'):
    (h, w) = src.shape
    (f_h, f_w) = filter.shape
    src_pad = my_padding(src, (f_h//2, f_w//2), pad_type)
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            val = np.sum(src_pad[row:row+f_h, col:col+f_w]* filter)
            dst[row, col] = val
    return dst