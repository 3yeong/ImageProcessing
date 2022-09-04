import cv2
import numpy as np

# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.filtering import my_filtering

def my_get_Gaussian2D_mask(msize, sigma=1):
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    
    gaus2Dx = -(x/sigma**2) * np.exp(-((x**2 + y**2)/(2*sigma**2)))
    gaus2Dy = -(y/sigma**2) * np.exp(-((x**2 + y**2)/(2*sigma**2)))
    return gaus2Dx, gaus2Dy


def get_DoG_filter(fsize, sigma=1):
    DoG_x ,DoG_y = my_get_Gaussian2D_mask(fsize, sigma)

    return DoG_x, DoG_y

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    DoG_x, DoG_y = get_DoG_filter(fsize=3, sigma=1)

    ###################################################
    # TODO                                            #
    # DoG mask sigma값 조절해서 mask 만들기              #
    ###################################################
    # DoG_x, DoG_y filter 확인
    x, y = get_DoG_filter(fsize=256, sigma=40)
    x = ((x-np.min(x))/np.max(x - np.min(x))*255).astype(np.uint8)
    y = ((y-np.min(y))/np.max(y - np.min(y))*255).astype(np.uint8)

    dst_x = my_filtering(src, DoG_x, 'zero')
    dst_y = my_filtering(src, DoG_y, 'zero')


    ###################################################
    # TODO                                            #
    # dst_x, dst_y 를 사용하여 magnitude 계산            #
    ###################################################
    dst = (np.abs(dst_x) + np.abs(dst_y))

    cv2.imshow('DoG_x filter', x)
    cv2.imshow('DoG_y filter', y)
    cv2.imshow('dst_x', dst_x/255)
    cv2.imshow('dst_y', dst_y/255)
    cv2.imshow('dst', dst/255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

