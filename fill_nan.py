# Original Matlab code https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
#
#
# Python port of depth filling code from NYU toolbox
# Speed needs to be improved
#
# Uses 'pypardiso'r solve
#
import scipy
import skimage
import numpy as np
from pypardiso import spsolve
from PIL import Image
from read_depth import depth_read
import time
import numba as nb

#
# fill_depth_colorization.m
# Preprocesses the kinect depth image using a gray scale version of the
# RGB image as a weighting for the smoothing. This code is a slight
# adaptation of Anat Levin's colorization code:
#
# See: www.cs.huji.ac.il/~yweiss/Colorization/
#
# Args:
#  imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
#      be between 0 and 1.
#  imgDepth - HxW matrix, the depth image for the current frame in
#       absolute (meters) space.
#  alpha - a penalty value between 0 and 1 for the current depth values.

def for_loop(W, H, winRad, rows, cols, gvals, indsM, grayImg, vals, len_, absImgNdx):
    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)

            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06

            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]

            # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1  # sum(gvals(1:nWin))

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    return W, H, winRad, rows, cols, gvals, indsM, grayImg, vals, len_, absImgNdx


def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
    t0 = time.time()
    imgIsNoise = imgDepthInput == 0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int)
    grayImg = skimage.color.rgb2gray(imgRgb)
    winRad = 1
    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window
    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1
    t1 = time.time()
    print('t1=%.2f' % (t1-t0))
    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)

            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06

            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]

            # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1  # sum(gvals(1:nWin))

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    t2 = time.time()
    print('t2=%.2f' % (t2-t1))
    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    # print ('Solving system..')

    new_vals = spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    # print ('Done.')

    denoisedDepthImg = new_vals * maxImgAbsDepth

    output = denoisedDepthImg.reshape((H, W)).astype('float32')

    output = np.multiply(output, (1 - knownValMask)) + imgDepthInput
    t3 = time.time()
    print('t3=%.2f' % (t3-t2))
    return output




def fill_depth_colorization_jit(imgRgb=None, imgDepthInput=None, alpha=1):
    t0 = time.time()
    imgIsNoise = imgDepthInput == 0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    indsM_copy = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int)
    grayImg = skimage.color.rgb2gray(imgRgb)
    winRad = 1
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    cols = []
    rows = []
    vals = []
    t1 = time.time()
    print('t1=%.2f' % (t1-t0))
    def func(x, W, H, winRad, len_window, indsM_copy, grayImg):

        i = x % H
        j = x // H
        nWin = 0
        gvals = np.zeros(len_window) - 1
        for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
            for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                if ii == i and jj == j:
                    continue

                rows.append(x)
                cols.append(indsM_copy[ii][jj])
                gvals[nWin] = grayImg[ii, jj]
                nWin = nWin + 1

        curVal = grayImg[i, j]
        gvals[nWin] = curVal
        c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)

        csig = c_var * 0.6
        mgv = np.min((gvals[:nWin] - curVal) ** 2)
        if csig < -mgv / np.log(0.01):
            csig = -mgv / np.log(0.01)

        if csig < 2e-06:
            csig = 2e-06

        gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
        gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
        for each in -gvals[:nWin]:
            vals.append(each)

        # Now the self-reference (along the diagonal).
        rows.append(x)
        cols.append(x)
        vals.append(1)  # sum(gvals(1:nWin))
        return x

    trans = np.vectorize(func, excluded=['indsM_copy', 'grayImg'])
    trans(indsM, W, H, winRad, len_window, indsM_copy=indsM_copy, grayImg=grayImg)

    t2 = time.time()
    print('t2=%.2f' % (t2-t1))

    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    # print ('Solving system..')

    new_vals = spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    # print ('Done.')

    denoisedDepthImg = new_vals * maxImgAbsDepth

    output = denoisedDepthImg.reshape((H, W)).astype('float32')

    output = np.multiply(output, (1 - knownValMask)) + imgDepthInput
    t3 = time.time()
    print('t3=%.2f' % (t3-t2))
    return output


if __name__ == '__main__':
    t1 = time.time()
    rgb_img = np.array(Image.open('0000000016_RGB.png'), dtype=int)
    rgb_img = rgb_img / np.max(rgb_img)
    depth_img = depth_read('0000000016.png')
    a = fill_depth_colorization_jit(rgb_img, depth_img)
    print(a.shape)
    print(np.min(a), np.max(a))
    t2 = time.time()
    print('Used time is %.2f' % (t2 - t1))
    a = (a/a.max() * 255).astype('int8')
    img = Image.fromarray(a)
    img.show()