import cv2
import numpy as np

def padding(src, ph, pw):
    (h, w) = src.shape

    dst = np.zeros((h+2*ph, w+2*pw))

    for row in range(ph, h+ph):
        for col in range(pw, w+ph):
            dst[row, col] = src[row-ph, col-pw]

    return dst

def dilation(B, S):
    (B_h, B_w) = B.shape
    (S_h, S_w) = S.shape
    ph, pw = S_h//2, S_w//2

    dst = padding(B, ph, pw)

    for row in range(B_h):
        for col in range(B_w):
            if B[row, col] == 1:
                dst[row:row+S_h, col:col+S_w] = S
    dst = dst[ph : B_h + ph, pw : B_w + pw]
    return dst

def erosion(B, S):
    (B_h, B_w) = B.shape
    (S_h, S_w) = S.shape
    ph, pw = S_h // 2, S_w // 2

    dst = np.zeros(B.shape)

    for row in range(ph, B_h-ph):
        for col in range(pw, B_w-pw):
            if np.array_equal(B[row-ph : row+ph+1, col-pw:col+pw+1], S):
                dst[row, col] = 1
    return dst

def opening(B, S):
    dst = dilation(erosion(B, S), S)
    return dst

def closing(B, S):
    dst = erosion(dilation(B, S), S)
    return dst


if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])


    cv2.imwrite('morphology_B.png', (B*255).astype(np.uint8))

    img_dilation = dilation(B, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)


