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

def my_filtering(src, ftype, fshape, pad_type='zero'):
    (h, w) = src.shape
    (m_h, m_w) = fshape
    src_pad = my_padding(src, (fshape[0], fshape[1]), pad_type)
    dst = np.zeros((h, w))

    if ftype == 'average':
        print('average filtering')
        ###################################################
        # TODO                                            #
        # mask 완성                                        #
        # 꼭 한줄로 완성할 필요 없음                           #
        ###################################################
        mask = np.ones(fshape)/(m_h * m_w)

        #mask 확인
        print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                       #
        # 꼭 한줄로 완성할 필요 없음                          #
        ##################################################
        mask = -1*np.ones(fshape)/(m_h * m_w)
        mask[m_h // 2][m_w // 2] = mask[m_h//2][m_w//2] + 2

        #mask 확인
        print(mask)

    #########################################################
    # TODO                                                  #
    # dst 완성                                               #
    # dst : filtering 결과 image                             #
    # 꼭 한줄로 완성할 필요 없음                                 #
    #########################################################
    for row in range(m_h, m_h + h):
        for col in range(m_w, m_w + w):

            dst[row - m_h, col - m_w] = np.sum(src_pad[row - (m_h//2) : row + (m_h//2)+1, col - (m_w//2) : col + (m_w//2) +1] * mask)

    dst = (dst+0.5).astype(np.uint8)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # repetition padding test
    rep_test = my_padding(src, (20,20))

    # 3x3 filter
    #dst_average = my_filtering(src, 'average', (3,3))
    #dst_sharpening = my_filtering(src, 'sharpening', (3,3))

    #원하는 크기로 설정
    #dst_average = my_filtering(src, 'average', (5,7))
    #dst_sharpening = my_filtering(src, 'sharpening', (7,5))

    # 11x13 filter
    dst_average = my_filtering(src, 'average', (11,13), 'repetition')
    dst_sharpening = my_filtering(src, 'sharpening', (11,13), 'repetition')

    cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    cv2.imshow('repetition padding test', rep_test.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
