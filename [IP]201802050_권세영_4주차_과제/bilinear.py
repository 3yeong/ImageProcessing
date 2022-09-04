import cv2
import numpy as np

def my_bilinear(src, scale):
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst))

    xline = np.linspace(0, h-1, h_dst) #src를 줄인 비율의 위치만큼 띄워서 배열에 저장
    yline = np.linspace(0, w-1, w_dst)

    # bilinear interpolation 적용
    for row in range(h_dst):
        for col in range(w_dst):
            # 참고로 꼭 한줄로 구현해야 하는건 아닙니다 여러줄로 하셔도 상관없습니다.(저도 엄청길게 구현했습니다.)

            u = xline[row]
            v = yline[col]

            m = int(u)
            n = int(v)

            t = u - m
            s = v - n

            if m+1 >= h and n+1 >= w:
                dst[row, col] = (1 - s) * (1 - t) * src[m, n] \
                                + s * (1 - t) * src[m, n] \
                                + (1 - s) * t * src[m, n] \
                                + s * t * src[m, n]

            elif m+1 >= h:
                dst[row, col] = (1 - s) * (1 - t) * src[m, n] \
                                + s * (1 - t) * src[m, n + 1] \
                                + (1 - s) * t * src[m, n] \
                                + s * t * src[m, n + 1]

            elif n+1 >= w:
                dst[row, col] = (1 - s) * (1 - t) * src[m, n] \
                                + s * (1 - t) * src[m, n] \
                                + (1 - s) * t * src[m + 1, n] \
                                + s * t * src[m + 1, n]
            else:
                dst[row, col] = (1-s) * (1-t) * src[m,n]\
                                +s * (1-t) * src[m, n+1]\
                                +(1-s) * t * src[m+1, n]\
                                +s * t * src[m+1, n+1]
    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    scale = 1/7
    #이미지 크기 1/2배로 변경
    my_dst_mini = my_bilinear(src, scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    #이미지 크기 2배로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, 1/scale)
    my_dst = my_dst.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


